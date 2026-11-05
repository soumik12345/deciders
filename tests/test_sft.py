from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deciders.models.kev import KevDecisionModelConfig
from deciders.train.sft import (
    DataloaderConfig,
    SupervisedTrainer,
    SupervisedTrainerConfig,
    TrackioConfig,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(2))

    def forward(self, inputs, *, temperature, return_logits):
        return self.projection(inputs) / temperature


class TinyTrainer(SupervisedTrainer):
    def create_dataloader(self):
        rows = [
            {
                "inputs": torch.tensor(inputs),
                "labels": label,
                "question_weights": weight,
            }
            for inputs, label, weight in (
                ([3.0, 0.0], 0, 0.5),
                ([0.0, 2.0], 0, 0.5),
                ([0.0, 1.0], 1, 1.0),
            )
        ]
        loader = DataLoader(rows, batch_size=2)
        return None, {"train": loader, "validation": loader}

    def create_model(self):
        return TinyModel()


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    run = Mock(id="test-run")
    monkeypatch.setattr("deciders.train.sft.trackio.init", Mock(return_value=run))
    monkeypatch.setattr("deciders.train.sft.trackio.Artifact", Mock())
    config = SupervisedTrainerConfig(
        dataloader_config=DataloaderConfig(dataset_name="test", batch_size=2),
        model_config=KevDecisionModelConfig(backbone_model_name_or_path="test"),
        trackio_config=TrackioConfig(project_name="test"),
        backend="eager",
        steps=3,
        validation_interval=2,
        checkpoint_interval=2,
        checkpoint_dir=tmp_path,
        optimizer_config={"name": "AdamW", "lr": 0.01},
        lr_scheduler_config={
            "name": "OneCycleLR",
            "max_lr": 0.01,
            "pct_start": 0.5,
            "div_factor": 1.0,
            "final_div_factor": 10.0,
            "cycle_momentum": False,
        },
    )
    trainer = TinyTrainer(config)
    yield trainer
    trainer.finish()


def test_train_step_updates_parameters_and_scheduler(trainer):
    batch = next(iter(trainer.train_loader))
    original = trainer.model.projection.weight.detach().clone()
    loss = F.cross_entropy(batch["inputs"], batch["labels"], reduction="none")
    weights = batch["question_weights"]

    metrics = trainer.train_step(batch)

    assert metrics["train/loss"] == pytest.approx((loss * weights).sum().item())
    assert metrics["train/accuracy_record"] == pytest.approx(0.5)
    assert metrics["train/accuracy_question"] == pytest.approx(0.5)
    assert metrics["train/grad_norm"] > 0
    assert metrics["train/learning_rate"] == pytest.approx(0.01)
    assert trainer.optimizer.param_groups[0]["lr"] < 0.01
    assert trainer.lr_scheduler.last_epoch == 1
    assert not torch.equal(original, trainer.model.projection.weight)
    assert "labels" in batch and "question_weights" in batch


def test_validation_weights_batches_and_restores_mode(trainer):
    inputs = torch.tensor([[3.0, 0.0], [0.0, 2.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    weights = torch.tensor([0.5, 0.5, 1.0])
    expected_loss = (
        F.cross_entropy(inputs, labels, reduction="none") * weights
    ).sum() / weights.sum()

    metrics = trainer.perform_validation(trainer.validation_loader)

    assert metrics["validation/loss"] == pytest.approx(expected_loss.item())
    assert metrics["validation/accuracy_record"] == pytest.approx(0.75)
    assert metrics["validation/accuracy_question"] == pytest.approx(2 / 3)
    assert trainer.model.training
    with pytest.raises(ValueError, match="no weighted questions"):
        trainer.perform_validation([])
    assert trainer.model.training
    trainer.model.eval()
    trainer.perform_validation(trainer.validation_loader)
    assert not trainer.model.training


def test_training_rolls_epochs_and_saves_restorable_checkpoint(trainer):
    run = trainer.run
    trainer.train()

    assert [call.kwargs["step"] for call in run.log.call_args_list] == [1, 2, 3]
    assert run.log.call_args_list[-1].args[0]["train/epoch"] == 1
    assert "validation/loss" in run.log.call_args_list[-1].args[0]
    checkpoint = torch.load(trainer.checkpoint_dir / "training.pt", weights_only=True)
    assert checkpoint["global_step"] == 3
    assert checkpoint["epoch"] == 1
    assert checkpoint["best_validation_loss"] < float("inf")
    assert not (trainer.checkpoint_dir / "training.pt.tmp").exists()

    restored = TinyModel()
    restored.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    scheduler_config = dict(trainer.config.lr_scheduler_config)
    scheduler_config.pop("name")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, total_steps=trainer.config.steps, **scheduler_config
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    torch.testing.assert_close(
        restored.projection.weight, trainer.model.projection.weight
    )
    assert scheduler.state_dict() == trainer.lr_scheduler.state_dict()
    restored_optimizer_state = next(iter(optimizer.state.values()))
    trainer_optimizer_state = next(iter(trainer.optimizer.state.values()))
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            restored_optimizer_state[key], trainer_optimizer_state[key]
        )
    assert run.log_artifact.call_args.kwargs["aliases"][-1] == "final"

    trainer.finish()
    trainer.finish()
    run.finish.assert_called_once()
