import json
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from torch.utils.data import DataLoader, DistributedSampler

from deciders.data import DataCollator, KevDataEncoder
from deciders.models.kev import KevDecisionModel, KevDecisionModelConfig
from deciders.train.sft import (
    DataloaderConfig,
    SupervisedTrainer,
    SupervisedTrainerConfig,
    TrackioConfig,
)

trackio_config = TrackioConfig(project_name="kev_0_8b_repro")
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
        "dtype": torch.float32,
        "attn_implementation": "sdpa",
    },
)
trainer_config = SupervisedTrainerConfig(
    dataloader_config=dataloader_config,
    model_config=model_config,
    trackio_config=trackio_config,
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
        return data_encoder, dataloaders

    def create_model(self) -> KevDecisionModel:
        return KevDecisionModel(self.config.model_config)


trainer = KevTrainer(config=trainer_config)
trainer.train()
trainer.finish()
