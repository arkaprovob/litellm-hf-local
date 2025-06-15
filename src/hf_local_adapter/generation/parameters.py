"""
Generation parameter management utilities.
"""

from typing import Dict, Any, Optional, List

from transformers import StoppingCriteriaList, PreTrainedTokenizer

from .stopping_criteria import StopOnTokens


class GenerationParameterManager:
    """Manages generation parameters and stopping criteria."""

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            default_params: Optional[Dict[str, Any]] = None,
            stopping_ids: Optional[List[int]] = None,
            eos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.default_params = default_params or {}

        # Set up stopping criteria
        self.stopping_criteria = self._setup_stopping_criteria(
            stopping_ids, eos_token_id, pad_token_id
        )

    def _setup_stopping_criteria(
            self,
            stopping_ids: Optional[List[int]] = None,
            eos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None
    ) -> StoppingCriteriaList:
        """Set up stopping criteria for generation."""
        stop_ids = stopping_ids or []

        # Add EOS token if specified
        if eos_token_id is not None:
            stop_ids.append(eos_token_id)
        elif self.tokenizer.eos_token_id is not None:
            stop_ids.append(self.tokenizer.eos_token_id)

        # Create stopping criteria
        if stop_ids:
            return StoppingCriteriaList([StopOnTokens(stop_ids)])
        else:
            return StoppingCriteriaList([])

    def prepare_generation_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Prepare generation kwargs by merging defaults with provided kwargs."""
        gen_kwargs = self.default_params.copy()

        # List of LiteLLM-specific parameters to exclude
        litellm_params = {"stream", "messages", "prompt", "model", "api_key", "api_base"}

        # Override with any provided kwargs (excluding LiteLLM-specific ones)
        for key in ["temperature", "top_p", "top_k", "max_new_tokens",
                    "repetition_penalty", "do_sample"]:
            if key in kwargs:
                gen_kwargs[key] = kwargs[key]

        # Add stopping criteria
        gen_kwargs["stopping_criteria"] = self.stopping_criteria

        # Set pad token ID for generation
        if "pad_token_id" not in gen_kwargs and self.tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        # Add any additional kwargs from optional_params (excluding LiteLLM params)
        optional_params = kwargs.get("optional_params", {})
        for key, value in optional_params.items():
            if key not in litellm_params:
                gen_kwargs[key] = value

        return gen_kwargs
