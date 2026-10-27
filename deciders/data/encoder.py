from __future__ import annotations

import re
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

import torch
from transformers import AutoTokenizer

from deciders.utils.data_utils import render_option_text, render_structured_content

if TYPE_CHECKING:
    from transformers.tokenization_utils_sentencepiece import SentencePieceBackend
    from transformers.tokenization_utils_tokenizers import TokenizersBackend


class DataEncoder:
    """Base encoder for encoding the inputs for implementing the encoding scheme
    for state, questions, and options.

    Args:
        tokenizer (TokenizersBackend | SentencePieceBackend): Tokenizer.
        max_state_tokens (int): State-token limit, including its opening marker.
            Defaults to Kev's training limit of 384.
        max_branch_tokens (int): Total row-token limit, including the state,
            question, options, and markers. Defaults to 1024.
        enforce_strict_encoding (bool): If True, reject an overlong state. Otherwise retain
            its first `max_state - 1` text tokens, matching Kev's default.
            An overlong total row is always rejected. Defaults to False.
    """

    def __init__(
        self,
        tokenizer: TokenizersBackend | SentencePieceBackend,
        max_state_tokens: int,
        max_branch_tokens: int,
        shuffle_choices: bool = True,
        enforce_strict_encoding: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_state_tokens = max_state_tokens
        self.max_branch_tokens = max_branch_tokens
        self.shuffle_choices = shuffle_choices
        self.enforce_strict_encoding = enforce_strict_encoding
        self.special_tokens, self.control_token_pattern = self.get_token_patterns()

    @abstractmethod
    def get_token_patterns(self) -> tuple[Sequence[str], re.Pattern]:
        raise NotImplementedError(
            (
                "Subclasses of DataEncoder should override this function to return ",
                "special_tokens (a list of special tokens) and control_token_pattern (a re.Pattern)",
            )
        )

    def encode_question(
        self,
        state: str,
        question: str,
        options: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode rendered text for a singe (state, question, options) without padding or labels.

        Args:
            state (str): Rendered state text.
            question (str): Rendered question text.
            options (list(str)): Rendered list of options.

        Returns:
            Long tensors: input IDs [1, length], closing-option
                positions [1, options], and the decision-marker position [1].
                The state limit includes its marker; the branch limit covers the
                entire row. Strict mode rejects overlong states; otherwise only
                state text is truncated. An overlong row is always rejected.
        """
        if not options:
            raise ValueError("Every example must have at least one option")

        marker_ids = []
        for token in self.special_tokens:
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
            escaped = self.control_token_pattern.sub(r"<¦\1¦>", text)
            return self.tokenizer.encode(escaped, add_special_tokens=False)

        state_tokens = user_ids(state)
        if (
            self.enforce_strict_encoding
            and len(state_tokens) + 1 > self.max_state_tokens
        ):
            raise ValueError(
                f"state exceeds {self.max_state_tokens} tokens: {len(state_tokens) + 1}"
            )
        ids = [state_id] + state_tokens[: self.max_state_tokens - 1]
        ids += [question_id] + user_ids(question)

        option_positions = []
        for option in options:
            ids += [option_id] + user_ids(option) + [close_id]
            option_positions.append(len(ids) - 1)
        ids.append(decide_id)

        if len(ids) > self.max_branch_tokens:
            raise ValueError(f"row exceeds {self.max_branch_tokens} tokens: {len(ids)}")

        return (
            torch.tensor([ids], dtype=torch.long, device="cpu"),
            torch.tensor([option_positions], dtype=torch.long, device="cpu"),
            torch.tensor([len(ids) - 1], dtype=torch.long, device="cpu"),
        )

    def encode_sequence(
        self, state: str, branches: dict[str, dict[str, Any]]
    ) -> list[dict[str, torch.Tensor]]:
        if not branches:
            raise ValueError("A record must contain at least one question.")
        state_text = render_structured_content(state)
        rows: list[dict[str, torch.Tensor]] = []
        for question_id, branch in branches.items():
            question_type = branch["type"]
            question_text = render_structured_content(branch.get("instructions"))
            criteria = branch.get("criteria")
            options: list[str]
            raw_label = branch["label"]
            if branch.get("target") is not None:
                raise ValueError(
                    f"{question_id}: soft targets require separate handling."
                )

            if question_type == "choice":
                if not isinstance(criteria, dict):
                    raise ValueError(
                        f"{question_id}: choice criteria must be a dictionary."
                    )
                keys = list(criteria)
                options = [render_option_text(key, criteria[key]) for key in keys]
                label = keys.index(raw_label)
                if self.shuffle_choices:
                    order = torch.randperm(len(options)).tolist()
                    options = [options[index] for index in order]
                    label = order.index(label)
            elif question_type == "score":
                if not isinstance(criteria, list):
                    raise ValueError(f"{question_id}: score criteria must be a list.")
                options = [render_structured_content(level) for level in criteria]
                label = int(raw_label)
            elif question_type == "noul":
                if criteria is None:
                    criteria = {}
                if not isinstance(criteria, dict):
                    raise ValueError(
                        f"{question_id}: noul criteria must be a dictionary or None."
                    )
                options = [
                    render_option_text("no", criteria.get("false")),
                    render_option_text("yes", criteria.get("true")),
                ]
                label = int(raw_label)
            else:
                raise ValueError(f"{question_id}: unsupported type {question_type!r}.")

            if not options or not 0 <= label < len(options):
                raise ValueError(f"{question_id}: invalid label or empty options.")

            input_ids, option_positions, decide_position = self.encode_question(
                state=state_text, question=question_text, options=options
            )

            rows.append(
                {
                    "input_ids": input_ids.squeeze(0),
                    "option_positions": option_positions.squeeze(0),
                    "decide_position": decide_position.squeeze(0),
                    "labels": torch.tensor(label, dtype=torch.long),
                }
            )

        return rows


class KevDataEncoder(DataEncoder):
    """`DataEncoder` for encoding the state, questions, and options according Kev's encoding scheme.

    Args:
        tokenizer (TokenizersBackend | SentencePieceBackend): Tokenizer.
        max_state_tokens (int): State-token limit, including its opening marker.
            Defaults to Kev's training limit of 384.
        max_branch_tokens (int): Total row-token limit, including the state,
            question, options, and markers. Defaults to 1024.
        enforce_strict_encoding (bool): If True, reject an overlong state. Otherwise retain
            its first `max_state - 1` text tokens, matching Kev's default.
            An overlong total row is always rejected. Defaults to False.
    """

    def __init__(
        self,
        tokenizer: TokenizersBackend | SentencePieceBackend,
        max_state_tokens: int = 384,
        max_branch_tokens: int = 1024,
        shuffle_choices: bool = True,
        enforce_strict_encoding: bool = True,
    ) -> None:
        super().__init__(
            tokenizer,
            max_state_tokens,
            max_branch_tokens,
            shuffle_choices,
            enforce_strict_encoding,
        )

    def get_token_patterns(self):
        special_tokens = (
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|fim_suffix|>",
        )
        control_token_patterns = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
        return special_tokens, control_token_patterns

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        revision: str | None = None,
        max_state_tokens: int = 384,
        max_branch_tokens: int = 1024,
        shuffle_choices: bool = True,
        enforce_strict_encoding: bool = True,
    ) -> KevDataEncoder:
        """Initialize a `KevDataEncoder` from a checkpoint.

        Args:
            model_name_or_path (str): The model name or path for initializing the tokenizer.
            max_state_tokens (int): State-token limit, including its opening marker.
                Defaults to Kev's training limit of 384.
            max_branch_tokens (int): Total row-token limit, including the state,
                question, options, and markers. Defaults to 1024.
            enforce_strict_encoding (bool): If True, reject an overlong state. Otherwise retain
                its first `max_state - 1` text tokens, matching Kev's default.
                An overlong total row is always rejected. Defaults to False.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
        if tokenizer is None:
            raise ValueError(
                f"Unable to initialize tokenizer from {model_name_or_path}"
            )
        return cls(
            tokenizer,
            max_state_tokens,
            max_branch_tokens,
            shuffle_choices,
            enforce_strict_encoding,
        )
