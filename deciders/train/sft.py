from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import trackio
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.utils.data import DistributedSampler

from deciders.models.kev import KevDecisionModel
from deciders.utils.train_utils import create_lr_scheduler, create_optimizer

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from deciders.data import DataEncoder
    from deciders.models.kev import KevDecisionModel, KevDecisionModelConfig


@dataclass
class DataloaderConfig:
    dataset_name: str
    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 0
    max_state_tokens: int = 384
    max_branch_tokens: int = 1024
    shuffle_choices: bool = True
    enforce_strict_encoding: bool = True
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackioConfig:
    project_name: str
    run_name: str | None = None
    group_name: str | None = None
    space_id: str | None = None
    private: bool | None = None


@dataclass
class SupervisedTrainerConfig:
    dataloader_config: DataloaderConfig
    model_config: KevDecisionModelConfig
    trackio_config: TrackioConfig
    seed: int = 42
    epochs: int = 1
    steps: int = 1
    backend: str = "tpu"
    enable_fsdp: bool = True
    validation_interval: int = 100
    max_grad_norm: float = 1.0
    optimizer_config: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "AdamW",
            "lr": 1e-4,
            "weight_decay": 0.01,
            "foreach": False,
            "fused": False,
        }
    )
    lr_scheduler_config: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "OneCycleLR",
            "max_lr": 1e-4,
            "pct_start": 0.1,
            "anneal_strategy": "cos",
            "div_factor": 25.0,
            "final_div_factor": 1e4,
            "cycle_momentum": True,
            "base_momentum": 0.85,
            "max_momentum": 0.95,
            "three_phase": False,
        }
    )


class SupervisedTrainer(ABC):
    def __init__(self, config: SupervisedTrainerConfig) -> None:
        self.config = config

        if self.config.enable_fsdp:
            if self.config.backend == "tpu":
                dist.init_process_group(backend="tpu_dist")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            mesh = DeviceMesh(
                device_type="tpu",
                mesh=torch.arange(world_size),
                mesh_dim_names=("fsdp",),
            )

        torch.manual_seed(self.config.seed)
        self.data_encoder, dataloaders = self.create_dataloader()
        if "train" not in dataloaders:
            raise ValueError("create_dataloader() must provide a 'train' dataloader.")
        self.train_loader = dataloaders["train"]
        self.validation_loader = dataloaders.get("validation")
        if self.validation_loader is None:
            self.validation_loader = dataloaders.get("val")

        self.model = self.create_model()
        self.model = self.model.shard(mesh) if self.config.enable_fsdp else self.model
        self.model = torch.compile(
            self.model, backend=self.config.backend, fullgraph=False, dynamic=False
        )

        self.device = next(self.model.parameters()).device
        self.optimizer = create_optimizer(
            (param for param in self.model.parameters() if param.requires_grad),
            self.config.optimizer_config,
        )
        self.lr_scheduler = create_lr_scheduler(
            self.optimizer,
            {**self.config.lr_scheduler_config, "total_steps": self.config.steps},
        )

        self.run = trackio.init(
            project=self.config.trackio_config.project_name,
            name=self.config.trackio_config.run_name,
            group=self.config.trackio_config.group_name,
            space_id=self.config.trackio_config.space_id,
            private=self.config.trackio_config.private,
            config=asdict(self.config),
        )

    @abstractmethod
    def create_dataloader(self) -> tuple[DataEncoder, dict[str, DataLoader]]:
        raise NotImplementedError(
            "Suclasses of the SFTTrainer must overrive SFTTrainer.create_dataloader"
        )

    @abstractmethod
    def create_model(self) -> KevDecisionModel:
        raise NotImplementedError(
            "Suclasses of the SFTTrainer must overrive SFTTrainer.create_model"
        )

    def clip_gradients(self) -> float:
        """Clip gradients in place and return their pre-clipping global L2 norm."""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=(
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            max_norm=self.config.max_grad_norm,
            norm_type=2.0,
            error_if_nonfinite=True,
            foreach=False,
        )

        # Under FSDP2, the returned norm can itself be a DTensor, so we need to
        # materialize the global scalar for logging on every rank.
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        return grad_norm.item()

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        batch = {key: value.to(self.device) for key, value in batch.items()}
        labels = batch.pop("labels")
        weights = batch.pop("question_weights")

        self.optimizer.zero_grad(set_to_none=True)

        logits = self.model(**batch, temperature=1.0, return_logits=True)

        per_question_loss = F.cross_entropy(logits.float(), labels, reduction="none")
        record_weight = weights.sum()
        loss = (per_question_loss * weights).sum() / record_weight

        lr_for_logging = self.optimizer.param_groups[0]["lr"]
        loss.backward()
        grad_norm = self.clip_gradients()
        self.optimizer.step()
        self.lr_scheduler.step()

        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct_predictions = predictions.eq(labels).float()

        return {
            "train/loss": loss.item(),
            "train/accuracy_record": (
                (correct_predictions * weights).sum() / record_weight
            ).item(),
            "train/accuracy_question": correct_predictions.mean().item(),
            "train/learning_rate": lr_for_logging,
            "train/grad_norm": grad_norm,
        }

    def train_step_fsdp(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        batch = {key: value.to(self.device) for key, value in batch.items()}
        labels = batch.pop("labels")
        weights = batch.pop("question_weights")

        self.optimizer.zero_grad(set_to_none=True)

        logits = self.model(**batch, temperature=1.0, return_logits=True)
        per_question_loss = F.cross_entropy(logits.float(), labels, reduction="none")
        local_loss_sum = (per_question_loss * weights).sum()
        local_weight = weights.sum()

        global_weight = local_weight.detach().clone()
        dist.all_reduce(global_weight, op=dist.ReduceOp.SUM)

        loss_for_gradient_computation = (
            local_loss_sum * dist.get_world_size() / global_weight
        )
        lr_for_logging = self.optimizer.param_groups[0]["lr"]
        loss_for_gradient_computation.backward()
        grad_norm = self.clip_gradients()
        self.optimizer.step()
        self.lr_scheduler.step()

        with torch.no_grad():
            correct_predictions = logits.argmax(dim=-1).eq(labels).float()
            stats = torch.stack(
                [
                    local_loss_sum.detach(),
                    (correct_predictions * weights).sum(),
                    correct_predictions.sum(),
                    local_weight,
                    weights.new_tensor(labels.numel()),
                ]
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

            loss_sum, record_correct, question_correct, weight, count = (
                stats.cpu().tolist()
            )

        return {
            "train/loss": loss_sum / weight,
            "train/accuracy_record": record_correct / weight,
            "train/accuracy_question": question_correct / count,
            "train/learning_rate": lr_for_logging,
            "train/grad_norm": grad_norm,
        }

    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        batch = {key: value.to(self.device) for key, value in batch.items()}
        labels = batch.pop("labels")
        weights = batch.pop("question_weights")
        logits = self.model(**batch, temperature=1.0, return_logits=True)
        per_question_loss = F.cross_entropy(logits.float(), labels, reduction="none")
        loss = (per_question_loss * weights).sum() / weights.sum()
        predictions = logits.argmax(dim=-1)
        correct_predictions = predictions.eq(labels).float()
        return {
            "validation/loss": loss.item(),
            "validation/accuracy_record": (
                (correct_predictions * weights).sum() / weights.sum()
            ).item(),
            "validation/accuracy_question": correct_predictions.mean().item(),
        }

    @torch.no_grad()
    def perform_validation(self, validation_loader: DataLoader) -> dict[str, float]:
        """Aggregate validation metrics locally and, with FSDP, across ranks.

        Reduce metric numerators and denominators once after the local loop,
        then compute global means. All FSDP ranks must participate with matching
        forward counts. The loader must avoid counting sampler-padding duplicates
        for exact dataset metrics. Restore the model's previous mode on exit.
        """
        validation_metrics = {
            "validation/loss": 0.0,
            "validation/accuracy_record": 0.0,
            "validation/accuracy_question": 0.0,
        }
        total_record_weight = 0.0
        total_questions = 0
        was_training = self.model.training
        self.model.eval()
        try:
            for batch in validation_loader:
                batch_metrics = self.validation_step(batch)
                record_weight = batch["question_weights"].sum().item()
                question_count = batch["labels"].numel()
                denominators = {
                    "validation/loss": record_weight,
                    "validation/accuracy_record": record_weight,
                    "validation/accuracy_question": question_count,
                }
                for name, denominator in denominators.items():
                    validation_metrics[name] += batch_metrics[name] * denominator
                total_record_weight += record_weight
                total_questions += question_count

            if self.config.enable_fsdp:
                totals = torch.tensor(
                    [
                        validation_metrics["validation/loss"],
                        validation_metrics["validation/accuracy_record"],
                        validation_metrics["validation/accuracy_question"],
                        total_record_weight,
                        total_questions,
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                (
                    validation_metrics["validation/loss"],
                    validation_metrics["validation/accuracy_record"],
                    validation_metrics["validation/accuracy_question"],
                    total_record_weight,
                    total_questions,
                ) = totals.cpu().tolist()

            if total_questions == 0 or total_record_weight <= 0:
                raise ValueError(
                    "The validation dataloader produced no weighted questions."
                )
            return {
                "validation/loss": validation_metrics["validation/loss"]
                / total_record_weight,
                "validation/accuracy_record": (
                    validation_metrics["validation/accuracy_record"]
                    / total_record_weight
                ),
                "validation/accuracy_question": (
                    validation_metrics["validation/accuracy_question"] / total_questions
                ),
            }
        finally:
            self.model.train(was_training)

    def train(self) -> None:
        """Run the configured number of optimizer steps, repeating the loader."""
        for name in ("steps", "validation_interval"):
            value = getattr(self.config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        validation_loader = self.validation_loader

        epoch = 0
        if isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)
        train_step_fn = (
            self.train_step_fsdp if self.config.enable_fsdp else self.train_step
        )
        train_iterator = iter(self.train_loader)
        self.model.train()
        for step in range(1, self.config.steps + 1):
            try:
                batch = next(train_iterator)
            except StopIteration:
                epoch += 1
                if isinstance(self.train_loader.sampler, DistributedSampler):
                    self.train_loader.sampler.set_epoch(epoch)
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            train_metrics = train_step_fn(batch)
            if validation_loader is not None and (
                step % self.config.validation_interval == 0 or step == self.config.steps
            ):
                self.perform_validation(validation_loader)

    def finish(self):
        if hasattr(self, "run"):
            self.run.finish()
        if self.config.enable_fsdp:
            dist.destroy_process_group()
