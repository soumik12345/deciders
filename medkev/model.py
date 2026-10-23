from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class MedKev(nn.Module):
    """Score answer options using a language-model backbone and a pointer head.

    The backbone encodes one question per sequence. The pointer head compares
    the decision-token representation with each closing option-token
    representation and returns unnormalized logits of shape
    `(batch_size, num_options)`.

    Args:
        model_name (str): Hugging Face model ID or local checkpoint directory
            used to load both the tokenizer and the causal language model.
            The model's base model must expose an integer `config.hidden_size`.
        pointer_dim (int): Dimension of the query and key projections used
            to score options. Defaults to 256.
        lora_config (LoraConfig | None): Optional LoRA configuration for the
            backbone. Must use `TaskType.FEATURE_EXTRACTION`, a positive rank,
            `lora_alpha = 2 * r`, `lora_dropout = 0.05`, and one of the supported
            Kev target-module sets. Defaults to None.
        model_kwargs (dict[str, Any]): Keyword arguments passed to
            `AutoModelForCausalLM.from_pretrained`. Defaults to BF16 weights
            and SDPA attention. A supplied dictionary replaces these defaults.

    Raises:
        TypeError: If `lora_config` has an invalid type or the backbone's
            `config.hidden_size` is not an integer.
        ValueError: If `lora_config` does not satisfy the Kev requirements.
    """

    def __init__(
        self,
        model_name: str,
        pointer_dim: int = 256,
        lora_config: LoraConfig | None = None,
        model_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
    ) -> None:
        super().__init__()
        self._validate_lora_config(lora_config)
        self.model_name = model_name
        self.pointer_dim = pointer_dim
        self.lora_config = lora_config

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        language_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.backbone = (
            get_peft_model(language_model.base_model, self.lora_config)
            if self.lora_config is not None
            else language_model.base_model
        )

        hidden_size: object = getattr(self.backbone.config, "hidden_size", None)
        if not isinstance(hidden_size, int):
            raise TypeError(
                f"Expected hidden_size to be int, got {type(hidden_size).__name__}"
            )

        self.query_projection = nn.Linear(hidden_size, self.pointer_dim)
        self.key_projection = nn.Linear(hidden_size, self.pointer_dim)
        self.scale = self.pointer_dim**-0.5

    def _validate_lora_config(self, lora_config: LoraConfig | None = None) -> None:
        if lora_config is None:
            return
        if not isinstance(lora_config, LoraConfig):
            raise TypeError("lora_config must be a LoraConfig or None")
        if lora_config.task_type != TaskType.FEATURE_EXTRACTION:
            raise ValueError(
                "lora_config.task_type must be 'FEATURE_EXTRACTION' for a backbone "
                "without a language-model head"
            )
        if (
            isinstance(lora_config.r, bool)
            or not isinstance(lora_config.r, int)
            or lora_config.r <= 0
        ):
            raise ValueError("lora_config.r must be a positive integer")
        if lora_config.lora_alpha != 2 * lora_config.r:
            raise ValueError(
                f"lora_config.lora_alpha must equal 2 * r ({2 * lora_config.r}) "
                "to match Kev"
            )
        if lora_config.lora_dropout != 0.05:
            raise ValueError("lora_config.lora_dropout must be 0.05 to match Kev")

        qv = {"q_proj", "v_proj"}
        attention = qv | {"k_proj", "o_proj"}
        dense = attention | {"gate_proj", "up_proj", "down_proj"}
        hybrid = {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}
        allowed_targets = (qv, attention, dense, attention | hybrid, dense | hybrid)
        targets = lora_config.target_modules
        if not isinstance(targets, (list, set)) or set(targets) not in allowed_targets:
            raise ValueError(
                "lora_config.target_modules must explicitly list a Kev target set: "
                "q_proj/v_proj; q_proj/k_proj/v_proj/o_proj; or those attention "
                "projections plus gate_proj/up_proj/down_proj. The latter two sets "
                "may also include all five hybrid projections: in_proj_qkv, "
                "in_proj_z, in_proj_a, in_proj_b, out_proj"
            )

    def _validate_forward_params(
        self,
        batch_size: int,
        input_ids: torch.Tensor,
        option_positions: torch.Tensor,
        decide_position: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length]")
        if option_positions.ndim != 2 or option_positions.shape[0] != batch_size:
            raise ValueError(
                "option_positions must have shape [batch_size, num_options]"
            )
        if option_positions.shape[1] == 0:
            raise ValueError("Every example must have at least one option")
        if decide_position.shape != (batch_size,):
            raise ValueError("decide_position must have shape [batch_size]")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

    def forward(
        self,
        input_ids: torch.Tensor,
        option_positions: torch.Tensor,
        decide_position: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits (batch_size, num_options) for a batch of questions per sequence.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_len), with dtype
                `torch.long`. Sequences may be right-padded.
            option_positions (torch.Tensor): Zero-based indices of closing option tokens,
                of shape `(batch_size, num_options)`, with dtype `torch.long`. Every example in
                a batch must have the same number of real options.
            decide_position (torch.Tensor): Zero-based index of the decision token in each
                sequence, of shape `(batch_size,)`, with dtype `torch.long`.
            attention_mask (torch.Tensor): Optional mask of shape `(batch_size, seq_len)`,
                with 1/True for real tokens and 0/False for padding.

        Returns:
            Unnormalized option logits of shape `(batch_size, num_options)`.
        """
        batch_size = input_ids.shape[0]
        self._validate_forward_params(
            batch_size, input_ids, option_positions, decide_position, attention_mask
        )

        # hiddent_state -> (batch_size, seq_len, hidden_size)
        hidden_state = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state

        # decide_position -> (batch_size,...) -> one decision-token position per example in batch
        decide_position = decide_position.to(hidden_state.device)

        # option_positions -> (batch_size, num_options)
        # Note: num_options can be variable for different batches, but in a single batch,
        # num_options has to be the same for all the examples
        option_positions = option_positions.to(hidden_state.device)

        # batch_indices -> (0, 1, 2, ..., batch_size)
        batch_indices = torch.arange(batch_size, device=hidden_state.device)

        # select the decision token for each example
        decision_representation = hidden_state[batch_indices, decide_position]

        # select the option token for each example
        option_representation = hidden_state[batch_indices[:, None], option_positions]

        # attend the decions with the options
        query = self.query_projection(decision_representation)
        key = self.key_projection(option_representation)
        logits = torch.bmm(key, query.unsqueeze(-1)).squeeze(-1) * self.scale

        return logits
