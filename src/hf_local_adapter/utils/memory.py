"""
Memory monitoring and management utilities.
"""

import torch
from typing import Optional, Dict, Any
from .logging import setup_logging

logger = setup_logging()


def get_gpu_memory_info() -> Dict[int, Dict[str, float]]:
    """
    Get memory information for all available GPUs.
    
    Returns:
        Dict mapping GPU index to memory info (allocated, reserved, total in GB)
    """
    if not torch.cuda.is_available():
        return {}
    
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        memory_info[i] = {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': total - reserved
        }
    
    return memory_info


def print_gpu_memory_usage(title: Optional[str] = None) -> None:
    """
    Print GPU memory usage for all available GPUs.
    
    Args:
        title: Optional title to print above the memory info
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available - no GPU memory to display")
        return
    
    if title:
        logger.info(f"\n=== {title} ===")
    else:
        logger.info("\n=== GPU Memory Usage ===")
    
    memory_info = get_gpu_memory_info()
    
    if not memory_info:
        logger.info("No GPUs found")
        return
    
    for gpu_id, info in memory_info.items():
        gpu_name = torch.cuda.get_device_properties(gpu_id).name
        logger.info(
            f"GPU {gpu_id} ({gpu_name}): "
            f"{info['allocated']:.2f}GB allocated, "
            f"{info['reserved']:.2f}GB reserved, "
            f"{info['free']:.2f}GB free / {info['total']:.2f}GB total "
            f"({info['reserved']/info['total']*100:.1f}% used)"
        )


def print_model_device_map(model: torch.nn.Module, title: Optional[str] = None) -> None:
    """
    Print the device mapping of a model's parameters.
    
    Args:
        model: The model to analyze
        title: Optional title to print above the device map
    """
    if title:
        logger.info(f"\n=== {title} ===")
    else:
        logger.info("\n=== Model Device Map ===")
    
    # Check if model has hf_device_map attribute (from HuggingFace)
    if hasattr(model, 'hf_device_map'):
        logger.info("HuggingFace Device Map:")
        for module_name, device in model.hf_device_map.items():
            logger.info(f"  {module_name}: {device}")
    else:
        # Manually check parameter devices
        device_map = {}
        for name, param in model.named_parameters():
            device = str(param.device)
            if device not in device_map:
                device_map[device] = []
            device_map[device].append(name)
        
        logger.info("Parameter Device Distribution:")
        for device, params in device_map.items():
            logger.info(f"  {device}: {len(params)} parameters")
            if len(params) <= 10:  # Show parameter names if not too many
                for param in params[:5]:
                    logger.info(f"    - {param}")
                if len(params) > 5:
                    logger.info(f"    ... and {len(params) - 5} more")


def get_memory_footprint_estimate(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Estimate the memory footprint of a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with memory estimates
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory based on parameter count and dtype
    param_memory = 0
    for param in model.parameters():
        param_memory += param.numel() * param.element_size()
    param_memory_gb = param_memory / 1024**3
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_memory_gb': param_memory_gb,
        'estimated_inference_memory_gb': param_memory_gb * 1.2,  # Add overhead estimate
        'estimated_training_memory_gb': param_memory_gb * 4.0,   # Add gradient/optimizer overhead
    }


def print_model_memory_footprint(model: torch.nn.Module, title: Optional[str] = None) -> None:
    """
    Print estimated memory footprint of a model.
    
    Args:
        model: The model to analyze
        title: Optional title to print above the memory info
    """
    if title:
        logger.info(f"\n=== {title} ===")
    else:
        logger.info("\n=== Model Memory Footprint ===")
    
    footprint = get_memory_footprint_estimate(model)
    
    logger.info(f"Total Parameters: {footprint['total_parameters']:,}")
    logger.info(f"Trainable Parameters: {footprint['trainable_parameters']:,}")
    logger.info(f"Parameter Memory: {footprint['parameter_memory_gb']:.2f} GB")
    logger.info(f"Estimated Inference Memory: {footprint['estimated_inference_memory_gb']:.2f} GB")
    logger.info(f"Estimated Training Memory: {footprint['estimated_training_memory_gb']:.2f} GB") 