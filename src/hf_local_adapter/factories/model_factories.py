"""
Factory functions for creating adapters with model-specific configurations.
"""

from ..adapter import HuggingFaceLocalAdapterV2
from ..formatting.templates import ChatTemplates


def create_llama2_adapter(
        model_id: str = "meta-llama/Llama-2-7b-chat-hf",
        **kwargs
) -> HuggingFaceLocalAdapterV2:
    """Create an adapter configured for Llama-2 models."""
    return HuggingFaceLocalAdapterV2(
        model_id=model_id,
        messages_to_prompt=ChatTemplates.llama2_formatter,
        **kwargs
    )


def create_mistral_adapter(
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.1",
        **kwargs
) -> HuggingFaceLocalAdapterV2:
    """Create an adapter configured for Mistral models."""
    return HuggingFaceLocalAdapterV2(
        model_id=model_id,
        messages_to_prompt=ChatTemplates.mistral_formatter,
        **kwargs
    )


def create_falcon_adapter(
        model_id: str = "tiiuae/falcon-7b-instruct",
        **kwargs
) -> HuggingFaceLocalAdapterV2:
    """Create an adapter configured for Falcon models."""
    return HuggingFaceLocalAdapterV2(
        model_id=model_id,
        messages_to_prompt=ChatTemplates.falcon_formatter,
        **kwargs
    )
