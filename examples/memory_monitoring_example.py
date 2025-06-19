"""
Example demonstrating GPU memory monitoring feature.

This script shows how the memory monitoring automatically displays:
1. Model memory footprint
2. Device mapping (which layers are on which devices)
3. GPU memory usage after model loading

Run with: python examples/memory_monitoring_example.py
"""

import torch
from src import HuggingFaceLocalAdapterV2, ModelConfig


def main():
    # Example 1: Enable memory monitoring (default behavior)
    print("=== Example 1: With Memory Monitoring (Default) ===")
    
    config_with_monitoring = ModelConfig(
        model_id="microsoft/DialoGPT-medium",  # Small model for demo
        device_map="auto",
        max_memory={0: "2GB", "cpu": "4GB"},  # Limit memory for demo
        torch_dtype=torch.float16,
        show_memory_usage=True  # Default is True
    )
    
    adapter_with_monitoring = HuggingFaceLocalAdapterV2(model_config=config_with_monitoring)
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Disable memory monitoring
    print("=== Example 2: Without Memory Monitoring ===")
    
    config_without_monitoring = ModelConfig(
        model_id="microsoft/DialoGPT-medium",
        device_map="auto", 
        torch_dtype=torch.float16,
        show_memory_usage=False  # Disable monitoring
    )
    
    adapter_without_monitoring = HuggingFaceLocalAdapterV2(model_config=config_without_monitoring)
    print("Model loaded without memory monitoring output.")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Manual memory monitoring
    print("=== Example 3: Manual Memory Monitoring ===")
    
    from src.hf_local_adapter.utils import (
        print_gpu_memory_usage,
        print_model_device_map,
        print_model_memory_footprint
    )
    
    # You can also call these functions manually anytime
    print_gpu_memory_usage("Current GPU Status")
    print_model_device_map(adapter_with_monitoring.model, "Manual Device Map Check")
    print_model_memory_footprint(adapter_with_monitoring.model, "Manual Memory Footprint")


if __name__ == "__main__":
    main() 