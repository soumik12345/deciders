from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from deciders.data.encoder import DataEncoder


class DataCollator:
    """Encode decision records into question rows and batch them with
    right padding, attention and option masks, labels, and per-record loss weights.

    Args:
        data_encoder (DataEncoder): Data encoder that encodes state, questions, and options.
    """

    def __init__(self, data_encoder: DataEncoder) -> None:
        self.data_encoder = data_encoder

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Flatten records into question rows and right-pad a CPU tensor batch.

        Args:
            records (list[dict[str, Any]]): Records from the dataset.

        Returns:
            input_ids/attention_mask [Q, L], option_positions/option_mask
            [Q, K], and decide_position/labels/question_weights [Q], where Q is
            the total question count. L and K are the batch's maximum sequence
            length and option count. Each question gets weight 1 / the number
            of questions in its record, for computing a mean loss per record.
        """
        if not records:
            raise ValueError("Cannot collate an empty batch.")

        pad_id = self.data_encoder.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.data_encoder.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must provide a PAD or EOS token for padding.")
        if isinstance(pad_id, bool) or not isinstance(pad_id, int):
            raise TypeError("Padding token ID must be a single integer.")

        rows: list[dict[str, torch.Tensor]] = []
        weights: list[float] = []
        for record in records:
            record_rows = self.data_encoder.encode_sequence(
                record["state"], record["questions"]
            )
            rows.extend(record_rows)
            weights.extend([1.0 / len(record_rows)] * len(record_rows))

        input_ids = pad_sequence(
            [row["input_ids"] for row in rows],
            batch_first=True,
            padding_value=pad_id,
        )
        lengths = torch.tensor(
            [row["input_ids"].numel() for row in rows],
            dtype=torch.long,
            device="cpu",
        )
        # Length-based masking also works when EOS is used as PAD.
        attention_mask = (
            torch.arange(input_ids.shape[1], device="cpu")[None, :] < lengths[:, None]
        ).long()

        # Padded positions must remain valid indices for the model's gather.
        option_positions = pad_sequence(
            [row["option_positions"] for row in rows],
            batch_first=True,
            padding_value=0,
        )
        option_counts = torch.tensor(
            [row["option_positions"].numel() for row in rows],
            dtype=torch.long,
            device="cpu",
        )
        option_mask = (
            torch.arange(option_positions.shape[1], device="cpu")[None, :]
            < option_counts[:, None]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "option_positions": option_positions,
            "decide_position": torch.stack([row["decide_position"] for row in rows]),
            "option_mask": option_mask,
            "labels": torch.stack([row["labels"] for row in rows]),
            "question_weights": torch.tensor(
                weights, dtype=torch.float32, device="cpu"
            ),
        }
