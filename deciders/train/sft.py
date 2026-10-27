from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import trackio

from deciders.models.kev import KevDecisionModel
from deciders.utils.train_utils import create_optimizer

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from deciders.data import DataEncoder


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
class SFTTrainerConfig:
    dataloader_config: DataloaderConfig
    trackio_config: TrackioConfig
    steps: int = 1
    validation_interval: int = 100
    optimizer_config: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "AdamW",
            "lr": 1e-3,
            "weight_decay": 0.01,
        }
    )


class SFTTrainer(ABC):
    def __init__(self, model: KevDecisionModel, config: SFTTrainerConfig) -> None:
        self.model = model
        self.config = config
        self.device = next(self.model.parameters()).device
        self.optimizer = create_optimizer(
            self.model.parameters(), self.config.optimizer_config
        )
        self.data_encoder, dataloaders = self.create_dataloader()
        if "train" not in dataloaders:
            raise ValueError("create_dataloader() must provide a 'train' dataloader.")
        self.train_loader = dataloaders["train"]
        self.validation_loader = dataloaders.get("validation")
        if self.validation_loader is None:
            self.validation_loader = dataloaders.get("val")

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

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        batch = {key: value.to(self.device) for key, value in batch.items()}
        labels = batch.pop("labels")
        weights = batch.pop("question_weights")
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(**batch, temperature=1.0, return_logits=True)
        per_question_loss = F.cross_entropy(logits.float(), labels, reduction="none")
        record_weight = weights.sum()
        loss = (per_question_loss * weights).sum() / record_weight
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct_predictions = predictions.eq(labels).float()
        return {
            "train/loss": loss.item(),
            "train/accuracy_record": (
                (correct_predictions * weights).sum() / record_weight
            ).item(),
            "train/accuracy_question": correct_predictions.mean().item(),
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
        """Aggregate validation metrics and restore the model's previous mode."""
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
        train_iterator = iter(self.train_loader)
        self.model.train()
        for step in range(1, self.config.steps + 1):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            self.train_step(batch)
            if validation_loader is not None and (
                step % self.config.validation_interval == 0 or step == self.config.steps
            ):
                self.perform_validation(validation_loader)
