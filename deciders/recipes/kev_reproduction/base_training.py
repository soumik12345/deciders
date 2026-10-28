"""
Run using:

```bash
uv run python deciders/recipes/kev_reproduction/base_training.py
```
"""

import json
from typing import Any

import torch
from datasets import load_dataset
from peft import TaskType
from peft.tuners.lora.config import LoraConfig
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AttentionInterface,
    AttentionMaskInterface,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from deciders.data import DataCollator, KevDataEncoder
from deciders.models.kev import KevDecisionModel, KevDecisionModelConfig
from deciders.models.ops import tpu_splash_attention
from deciders.train.sft import (
    DataloaderConfig,
    SupervisedTrainer,
    SupervisedTrainerConfig,
    TrackioConfig,
)

AttentionInterface.register("tpu_splash", tpu_splash_attention)
AttentionMaskInterface.register(
    "tpu_splash", lambda *args, attention_mask=None, **kwargs: attention_mask
)


def decode_records(batch: dict[str, list[str]]) -> dict[str, list[Any]]:
    records = [json.loads(text) for text in batch["text"]]
    return {
        "state": [record["state"] for record in records],
        "questions": [record["questions"] for record in records],
    }


class KevTrainer(SupervisedTrainer):
    def create_dataloader(self) -> tuple[KevDataEncoder, dict[str, DataLoader]]:
        config = self.config.dataloader_config
        data_encoder = KevDataEncoder.from_pretrained(
            self.config.model_config.backbone_model_name_or_path,
            revision=self.config.model_config.revision,
            max_state_tokens=config.max_state_tokens,
            max_branch_tokens=config.max_branch_tokens,
            shuffle_choices=config.shuffle_choices,
            enforce_strict_encoding=config.enforce_strict_encoding,
        )
        validation_encoder = KevDataEncoder(
            data_encoder.tokenizer,
            max_state_tokens=config.max_state_tokens,
            max_branch_tokens=config.max_branch_tokens,
            shuffle_choices=False,
            enforce_strict_encoding=config.enforce_strict_encoding,
        )
        base_path = f"hf://datasets/{config.dataset_name}/v7/decision-v7"
        dataset_kwargs = dict(config.dataset_kwargs)
        dataset_kwargs.setdefault(
            "data_files",
            {
                "train": f"{base_path}/train.jsonl",
                "validation": f"{base_path}/development.jsonl",
                # "calibration": f"{base_path}/calibration.jsonl",
                # "test": f"{base_path}/test.jsonl",
            },
        )
        datasets = load_dataset("text", **dataset_kwargs)
        dataloaders = {}
        for split, dataset in datasets.items():
            # Using 200 validation samples to validate on the fly, to speed up training
            if split == "validation":
                dataset = dataset.select(range(min(200, len(dataset))))
            dataset = dataset.with_transform(decode_records)
            is_training = split == "train"
            sampler = None
            if is_training and self.config.enable_fsdp:
                sampler = DistributedSampler(
                    dataset, shuffle=config.shuffle, seed=self.config.seed
                )
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=is_training and config.shuffle and sampler is None,
                sampler=sampler,
                num_workers=config.num_workers,
                collate_fn=DataCollator(
                    data_encoder if is_training else validation_encoder
                ),
            )
        self.config.steps = self.config.epochs * len(dataloaders["train"])
        return data_encoder, dataloaders

    def create_model(self) -> KevDecisionModel:
        model = KevDecisionModel(self.config.model_config)
        for module in model.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.forward = torch.compiler.disable(module.forward, recursive=True)  # type: ignore
        return model


if __name__ == "__main__":
    trackio_config = TrackioConfig(
        project_name="deciders",
        run_name="kev_0_8b_repro_base",
        group_name="kev_0_8b_repro",
        space_id="geekyrakshit/deciders-trackio",
    )
    dataloader_config = DataloaderConfig(dataset_name="jaredpalmer/kev-suites")
    model_config = KevDecisionModelConfig(
        backbone_model_name_or_path="Qwen/Qwen3.5-0.8B-Base",
        pointer_dim=256,
        lora_config=LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                # Full-attention projections
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                # MLP projections
                "gate_proj",
                "up_proj",
                "down_proj",
                # Gated DeltaNet projections
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_a",
                "in_proj_b",
                "out_proj",
            ],
        ),
        model_kwargs={
            "dtype": torch.bfloat16,
            "attn_implementation": "tpu_splash",
        },
    )
    trainer_config = SupervisedTrainerConfig(
        dataloader_config=dataloader_config,
        model_config=model_config,
        trackio_config=trackio_config,
        epochs=2,
    )
    trainer = KevTrainer(config=trainer_config)
    try:
        trainer.train()
    finally:
        trainer.finish()
