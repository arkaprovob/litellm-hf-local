"""
Tokenization utilities and helpers.
"""

from typing import Dict

import torch
from transformers import PreTrainedTokenizer


class TokenizationUtils:
    """Utility class for tokenization operations."""

    def __init__(self, tokenizer: PreTrainedTokenizer, context_window: int = 4096):
        self.tokenizer = tokenizer
        self.context_window = context_window

    def tokenize_prompt(self, prompt: str, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """Tokenize prompt and prepare inputs for model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_window
        )

        # Move to device if specified
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        return inputs

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def validate_context_window(self) -> int:
        """Validate and adjust context window based on model configuration."""
        try:
            model_max_length = self.tokenizer.model_max_length
            if model_max_length and model_max_length < self.context_window:
                return model_max_length
        except Exception:
            pass
        return self.context_window
