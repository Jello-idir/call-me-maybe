"""Editor type hints for the school's llm_sdk package.

Keep these signatures aligned with llm_sdk/llm_sdk/__init__.py.
This file describes the API; the SDK's Python files provide the implementation.
"""

import torch


class Small_LLM_Model:
    """Load a Hugging Face causal language model and access its tokenizer and logits."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        """Load the model, automatically choosing the device and dtype when omitted."""
        ...

    def encode(self, text: str) -> torch.Tensor:
        """Tokenize text into a 2-D tensor of token IDs on the model's device."""
        ...

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Convert token IDs to text, removing special tokens."""
        ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return raw scores for each possible next token, without applying softmax."""
        ...

    def get_path_to_vocab_file(self) -> str:
        """Return the local vocabulary file path, downloading the file if needed."""
        ...

    def get_path_to_merges_file(self) -> str:
        """Return the local tokenizer merges file path, downloading it if needed."""
        ...

    def get_path_to_tokenizer_file(self) -> str:
        """Return the local tokenizer JSON file path, downloading it if needed."""
        ...
