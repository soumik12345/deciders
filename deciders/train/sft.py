from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import trackio
from tqdm.auto import tqdm

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
    use_nota_distraction: bool = True
    nota_distraction_proba: float = 0.5
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
    validation_interval: int = 100
    checkpoint_interval: int = 100
    checkpoint_dir: Path | str | None = None
    max_grad_norm: float = 1.0
    optimizer_config: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "AdamW",
            "lr": 1e-4,
            "weight_decay": 0.01,
            "foreach": False,
            "fused": True,
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

        torch.manual_seed(self.config.seed)
        self.data_encoder, dataloaders = self.create_dataloader()
        if "train" not in dataloaders:
            raise ValueError("create_dataloader() must provide a 'train' dataloader.")
        self.train_loader = dataloaders["train"]
        self.validation_loader = dataloaders.get("validation")
        if self.validation_loader is None:
            self.validation_loader = dataloaders.get("val")

        self.model = self.create_model()
        self.model.compile(backend=self.config.backend, fullgraph=False, dynamic=False)

        self.device = next(self.model.parameters()).device
        self.optimizer = create_optimizer(
            (param for param in self.model.parameters() if param.requires_grad),
            self.config.optimizer_config,
        )
        self.lr_scheduler = create_lr_scheduler(
            self.optimizer,
            {**self.config.lr_scheduler_config, "total_steps": self.config.steps},
        )

        self.checkpoint_dir = Path(
            self.config.checkpoint_dir
            if self.config.checkpoint_dir is not None
            else "./checkpoint_dir"
        )

        self.initialize_trackio()

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

    def initialize_trackio(self):
        run_config = asdict(self.config)
        run_config["checkpoint_dir"] = str(self.checkpoint_dir)
        run_config["total_batch_size"] = self.config.dataloader_config.batch_size
        self.run = trackio.init(
            project=self.config.trackio_config.project_name,
            name=self.config.trackio_config.run_name,
            group=self.config.trackio_config.group_name,
            space_id=self.config.trackio_config.space_id,
            private=self.config.trackio_config.private,
            config=run_config,
            auto_log_gpu=self.config.backend != "cuda",
        )

    def clip_gradients(self) -> float:
        """Clip gradients in place and return their pre-clipping L2 norm."""
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
        """Aggregate validation metrics over the full validation loader.

        Weight record and question metrics by their respective counts and
        restore the model's previous mode on exit.
        """
        validation_metrics = {
            "validation/loss": 0.0,
            "validation/accuracy_record": 0.0,
            "validation/accuracy_question": 0.0,
        }
        total_record_weight = 0.0
        total_questions = 0
        progress = tqdm(
            validation_loader,
            desc="Validation",
            unit="batch",
            leave=False,
            mininterval=1.0,
        )
        was_training = self.model.training
        self.model.eval()
        try:
            for batch in progress:
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
                if total_record_weight > 0:
                    running_loss = (
                        validation_metrics["validation/loss"] / total_record_weight
                    )
                    progress.set_postfix_str(f"loss={running_loss:.4f}", refresh=False)

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
            progress.close()
            self.model.train(was_training)

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        best_loss: float,
        artifact_name: str,
        aliases: list[str],
    ) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_dir / "training.pt.tmp"
        checkpoint = self.checkpoint_dir / "training.pt"

        torch.save(
            {
                "format_version": 1,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.lr_scheduler.state_dict(),
                "global_step": step,
                "epoch": epoch,
                "best_validation_loss": best_loss,
            },
            temporary,
        )
        temporary.replace(checkpoint)

        if self.run is not None:
            artifact = trackio.Artifact(
                name=artifact_name,
                type="checkpoint",
                metadata={
                    "global_step": step,
                    "epoch": epoch,
                },
            )
            artifact.add_file(checkpoint)
            self.run.log_artifact(artifact, aliases=aliases)

    def train(self) -> None:
        """Run optimizer steps with nested training and validation progress bars."""
        for name in ("steps", "validation_interval", "checkpoint_interval"):
            value = getattr(self.config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        validation_loader = self.validation_loader

        epoch = 0
        best_loss = float("inf")
        train_iterator = iter(self.train_loader)
        self.model.train()
        progress = tqdm(
            range(1, self.config.steps + 1),
            desc="Training",
            unit="step",
            mininterval=1.0,
        )
        for step in progress:
            try:
                batch = next(train_iterator)
            except StopIteration:
                epoch += 1
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            train_metrics = self.train_step(batch)
            metrics = {**train_metrics, "train/epoch": epoch}
            if validation_loader is not None and (
                step % self.config.validation_interval == 0 or step == self.config.steps
            ):
                validation_metrics = self.perform_validation(validation_loader)
                metrics.update(validation_metrics)

            validation_loss = metrics.get("validation/loss")
            is_best = validation_loss is not None and validation_loss < best_loss
            if validation_loss is not None and is_best:
                best_loss = validation_loss
            is_final = step == self.config.steps
            if step % self.config.checkpoint_interval == 0 or is_best or is_final:
                aliases = []
                if is_best:
                    aliases.append("best")
                if is_final:
                    aliases.append("final")
                self.save_checkpoint(
                    step=step,
                    epoch=epoch,
                    best_loss=best_loss,
                    artifact_name=(
                        f"checkpoint-{self.run.id}"
                        if self.run is not None
                        else "checkpoint"
                    ),
                    aliases=aliases,
                )

            progress_metrics = (
                f"epoch={epoch + 1} "
                f"loss={train_metrics['train/loss']:.4f} "
                f"lr={train_metrics['train/learning_rate']:.2e}"
            )
            if "validation/loss" in metrics:
                progress_metrics += f" val_loss={metrics['validation/loss']:.4f}"
            progress.set_postfix_str(progress_metrics, refresh=False)

            if self.run is not None:
                self.run.log(metrics, step=step)
        progress.close()

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
