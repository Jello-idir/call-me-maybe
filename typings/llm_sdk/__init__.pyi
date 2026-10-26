"""Editor type hints for the school's llm_sdk package.

Keep these signatures aligned with llm_sdk/llm_sdk/__init__.py.
This file describes the API; the SDK's Python files provide the implementation.
"""

import torch


class Small_LLM_Model:
    """Wrap a causal language model for token encoding and next-token scores."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        """Load the tokenizer and model, then configure inference.

        Args:
            model_name: Hugging Face model ID; defaults to Qwen/Qwen3-0.6B.
            device: Device name, or None to prefer MPS, then CUDA, then CPU.
            dtype: Weight precision, or None for float16 on MPS/CUDA and
                float32 on CPU.
            trust_remote_code: Whether to allow the model's custom code.
        """
        ...

    def encode(self, text: str) -> torch.Tensor:
        """Encode text without adding special tokens.

        Args:
            text: Text to tokenize.

        Returns:
            A tensor of shape (1, token_count) on the model's device.
        """
        ...

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Decode one token sequence while removing special tokens.

        Args:
            ids: A one-dimensional tensor or list of token IDs.

        Returns:
            Decoded text without special tokens.
        """
        ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Score every possible next token for a context.

        Args:
            input_ids: Context token IDs to pass to the model.

        Returns:
            Raw next-token scores indexed by token ID, without softmax.
        """
        ...

    def get_path_to_vocab_file(self) -> str:
        """Find or download the model's vocabulary file.

        Returns:
            Local path to the vocabulary JSON file.
        """
        ...

    def get_path_to_merges_file(self) -> str:
        """Find or download the model's tokenizer merges file.

        Returns:
            Local path to the merges text file.
        """
        ...

    def get_path_to_tokenizer_file(self) -> str:
        """Find or download the model's tokenizer definition.

        Returns:
            Local path to the tokenizer JSON file.
        """
        ...
