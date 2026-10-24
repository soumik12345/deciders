import re
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

_KEV_SPECIAL_TOKENS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)
_CONTROL_TOKEN_PATTERN = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class KevDecisionModel(nn.Module):
    """Score answer options using a language-model backbone and a pointer head.

    The backbone encodes one question per sequence. The pointer head compares
    the decision-token representation with each closing option-token
    representation and returns unnormalized logits as a padded tensor or
    nested lists of records and questions.

    The pointer head stays in FP32 and scores FP32 copies of the selected
    backbone outputs, matching Kev. Move the model with `.to(device)` without
    a dtype to preserve the backbone/head precision choices.

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

        self.query_projection = nn.Linear(
            hidden_size, self.pointer_dim, dtype=torch.float32
        )
        self.key_projection = nn.Linear(
            hidden_size, self.pointer_dim, dtype=torch.float32
        )
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
        option_mask: torch.Tensor | None = None,
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
        if option_mask is not None:
            if option_mask.shape != option_positions.shape:
                raise ValueError(
                    "option_mask must have the same shape as option_positions"
                )
            if option_mask.dtype != torch.bool:
                raise TypeError("option_mask must have dtype torch.bool")
            if not option_mask.any(dim=-1).all().item():
                raise ValueError("Every question must have at least one real option")

    def encode(
        self,
        state: str,
        question: str,
        options: list[str],
        *,
        max_state: int = 384,
        max_branch: int = 1024,
        strict: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one question as a causal row using Kev's delimiter tokens.

        Args:
            state (str): Shared context for the question.
            question (str): Question instructions.
            options (list[str]): Nonempty list of answer options in scoring order.
            max_state (int): State-token limit, including its opening marker.
                Defaults to Kev's training limit of 384.
            max_branch (int): Total row-token limit, including the state,
                question, options, and markers. Defaults to 1024.
            strict (bool): If True, reject an overlong state. Otherwise retain
                its first `max_state - 1` text tokens, matching Kev's default.
                An overlong total row is always rejected. Defaults to False.

        Returns:
            CPU tensors `(input_ids, option_positions, decide_position)` with
            dtype `torch.long` and shapes `(1, seq_len)`, `(1, num_options)`,
            and `(1,)`. Readout positions point to the exact closing-option
            and decision markers. No padding or extra BOS/EOS tokens are added.

        Raises:
            ValueError: If options are empty, limits are invalid, the tokenizer
                lacks Kev's single-token delimiters, or a context limit is
                exceeded under the selected policy.

        Notes:
            Control-token-like strings in user text are escaped before
            tokenization. Move the returned tensors to the model's device
            before passing them to `forward`.
        """
        if not options:
            raise ValueError("Every example must have at least one option")
        for name, limit in (("max_state", max_state), ("max_branch", max_branch)):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError(f"{name} must be a positive integer")

        marker_ids = []
        for token in _KEV_SPECIAL_TOKENS:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if (
                token_id is None
                or token_id == self.tokenizer.unk_token_id
                or self.tokenizer.encode(token, add_special_tokens=False) != [token_id]
            ):
                raise ValueError(
                    f"Tokenizer must provide Kev delimiter {token!r} as a single token"
                )
            marker_ids.append(token_id)
        state_id, question_id, option_id, close_id, decide_id = marker_ids

        def user_ids(text: str) -> list[int]:
            escaped = _CONTROL_TOKEN_PATTERN.sub(r"<¦\1¦>", text)
            return self.tokenizer.encode(escaped, add_special_tokens=False)

        state_tokens = user_ids(state)
        if strict and len(state_tokens) + 1 > max_state:
            raise ValueError(
                f"state exceeds {max_state} tokens: {len(state_tokens) + 1}"
            )
        ids = [state_id] + state_tokens[: max_state - 1]
        ids += [question_id] + user_ids(question)
        option_positions = []
        for option in options:
            ids += [option_id] + user_ids(option) + [close_id]
            option_positions.append(len(ids) - 1)
        ids.append(decide_id)

        if len(ids) > max_branch:
            raise ValueError(f"row exceeds {max_branch} tokens: {len(ids)}")

        return (
            torch.tensor([ids], dtype=torch.long, device="cpu"),
            torch.tensor([option_positions], dtype=torch.long, device="cpu"),
            torch.tensor([len(ids) - 1], dtype=torch.long, device="cpu"),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        option_positions: torch.Tensor,
        decide_position: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        option_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        return_logits: bool = True,
        questions_per_record: list[int] | None = None,
    ) -> torch.Tensor | list[list[torch.Tensor]]:
        """Score a flattened batch containing one question per sequence.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_len), with dtype
                `torch.long`. Sequences may be right-padded.
            option_positions (torch.Tensor): Zero-based indices of closing option tokens,
                of shape `(batch_size, max_options)`, with dtype `torch.long`.
                Pad with a valid token index, such as 0, and use option_mask
                when questions have different numbers of real options.
            decide_position (torch.Tensor): Zero-based index of the decision token in each
                sequence, of shape `(batch_size,)`, with dtype `torch.long`.
            attention_mask (torch.Tensor): Optional mask of shape `(batch_size, seq_len)`,
                with 1/True for real tokens and 0/False for padding.
            option_mask (torch.Tensor): Boolean tensor matching option_positions. `True` marks a
                real option and `False` marks padding. `None` means every option is real.
            temperature (float): Temperature applied to normalize the raw logits. During training,
                it should be `1.0`, i.e, raw unnormalized logits should be used.
                The temperature should be calibrated during inference or evaluation.
            return_logits (bool): If True, return a padded logits tensor.
                If False, return logits grouped by record and question, with
                padded options removed. Both formats contain unnormalized logits.
            questions_per_record (list[int] | None): Number of consecutive batch
                rows belonging to each record, in record order. Required when
                return_logits=False. Entries must be nonnegative integers summing
                to batch_size; a zero produces an empty question list.

        Returns:
            Temperature-scaled logits: either a tensor of shape
            `(batch_size, max_options)` with padded options set to negative infinity,
            or nested lists `records[record_index][question_index]` containing
            tensors of shape `(num_real_options,)`. Gradients are preserved.
        """
        batch_size = input_ids.shape[0]
        self._validate_forward_params(
            batch_size,
            input_ids,
            option_positions,
            decide_position,
            attention_mask,
            option_mask,
        )
        if not return_logits and questions_per_record is None:
            raise ValueError("questions_per_record is required for grouped output")
        if questions_per_record is not None:
            if not isinstance(questions_per_record, list) or any(
                isinstance(count, bool) or not isinstance(count, int)
                for count in questions_per_record
            ):
                raise TypeError("questions_per_record must be a list of integers")
            if any(count < 0 for count in questions_per_record):
                raise ValueError("questions_per_record entries must be nonnegative")
            if sum(questions_per_record) != batch_size:
                raise ValueError("questions_per_record must sum to batch_size")

        # hiddent_state -> (batch_size, seq_len, hidden_size)
        hidden_state = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state

        # decide_position -> (batch_size,...) -> one decision-token position per example in batch
        decide_position = decide_position.to(hidden_state.device)

        # option_positions -> (batch_size, num_options)
        option_positions = option_positions.to(hidden_state.device)

        # batch_indices -> (0, 1, 2, ..., batch_size)
        batch_indices = torch.arange(batch_size, device=hidden_state.device)

        # Cast only the selected representations; the pointer head stays FP32.
        decision_representation = hidden_state[batch_indices, decide_position].float()

        # select the option token for each example
        option_representation = hidden_state[
            batch_indices[:, None], option_positions
        ].float()

        # attend the decions with the options
        query = self.query_projection(decision_representation)
        key = self.key_projection(option_representation)
        logits = torch.bmm(key, query.unsqueeze(-1)).squeeze(-1) * self.scale
        logits = logits / temperature

        # Exclude padded option readouts from probabilities and loss.
        if option_mask is not None:
            option_mask = option_mask.to(logits.device)
            logits = logits.masked_fill(~option_mask, float("-inf"))

        if not return_logits:
            question_logits = (
                list(logits.unbind(dim=0))
                if option_mask is None
                else [row[mask] for row, mask in zip(logits, option_mask)]
            )

            assert questions_per_record is not None  # Validated before backbone execution.
            records = []
            offset = 0
            for count in questions_per_record:
                records.append(question_logits[offset : offset + count])
                offset += count

            return records

        return logits
