"""
HuggingFace Local Adapter Package

A well-organized package for running HuggingFace models locally through LiteLLM.
Provides clean separation of concerns with modular components.
"""

from .hf_local_adapter import HuggingFaceLocalAdapterV2
from .hf_local_adapter.config.model_config import ModelConfig
from .hf_local_adapter.factories.model_factories import (
    create_llama2_adapter,
    create_mistral_adapter,
    create_falcon_adapter
)

__version__ = "2.0.0"
__all__ = [
    "HuggingFaceLocalAdapterV2",
    "ModelConfig",
    "create_llama2_adapter",
    "create_mistral_adapter",
    "create_falcon_adapter"
]
