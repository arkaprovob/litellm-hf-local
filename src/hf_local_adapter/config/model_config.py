"""
Model configuration dataclass and related utilities.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any

import torch


@dataclass
class ModelConfig:
    """Configuration for HuggingFace model initialization."""
    model_id: str
    device: str = "cuda:0"
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: Optional[Union[str, Dict]] = "auto"
    max_memory: Optional[Dict[str, str]] = None
    offload_folder: Optional[str] = None
    revision: Optional[str] = None

    # Quantization config
    quantization_config: Optional[Dict[str, Any]] = None

    # Model-specific kwargs
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Tokenizer-specific kwargs
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
