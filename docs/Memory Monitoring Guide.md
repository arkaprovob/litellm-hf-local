# Memory Monitoring Guide

The litellm-hf-local adapter includes GPU memory monitoring that automatically displays detailed information when models are loaded.

## 🔍 What Gets Displayed

When you load a model, you'll automatically see:

1. **Model Memory Footprint** - Parameter count and memory estimates
2. **Device Mapping** - Which layers are on which devices
3. **GPU Memory Usage** - Real-time VRAM usage per GPU

## 📖 Basic Usage

### Automatic Monitoring (Default)

```python
from src import HuggingFaceLocalAdapterV2, ModelConfig

# Memory monitoring is enabled by default
config = ModelConfig(
    model_id="microsoft/Phi-4-reasoning",
    device_map="auto",
    max_memory={0: "10GB", 1: "10GB"}
)

# This will automatically display memory info
adapter = HuggingFaceLocalAdapterV2(model_config=config)
```

**Output Example:**
```
=== Model Memory Footprint: microsoft/Phi-4-reasoning ===
Total Parameters: 14,701,875,200
Trainable Parameters: 14,701,875,200
Parameter Memory: 27.35 GB
Estimated Inference Memory: 32.82 GB
Estimated Training Memory: 109.39 GB

=== Device Mapping: microsoft/Phi-4-reasoning ===
HuggingFace Device Map:
  model.embed_tokens: 0
  model.layers.0: 0
  model.layers.1: 0
  ...
  model.layers.30: 1
  model.layers.31: 1
  model.norm: 1
  lm_head: 1

=== GPU Memory After Loading: microsoft/Phi-4-reasoning ===
GPU 0 (NVIDIA GeForce RTX 4090): 9.23GB allocated, 9.87GB reserved, 14.13GB free / 24.00GB total (41.1% used)
GPU 1 (NVIDIA GeForce RTX 4090): 8.91GB allocated, 9.45GB reserved, 14.55GB free / 24.00GB total (39.4% used)
```

### Disable Automatic Monitoring

```python
config = ModelConfig(
    model_id="microsoft/Phi-4-reasoning", 
    show_memory_usage=False  # Disable automatic monitoring
)

adapter = HuggingFaceLocalAdapterV2(model_config=config)
# No memory info will be displayed
```

## 🛠️ Manual Memory Monitoring

You can call the monitoring functions manually at any time:

```python
from src.hf_local_adapter.utils import (
    print_gpu_memory_usage,
    print_model_device_map,
    print_model_memory_footprint
)

# Check GPU memory anytime
print_gpu_memory_usage("After Generation")

# Check model device distribution
print_model_device_map(adapter.model, "Current Device Map")

# Check model memory footprint
print_model_memory_footprint(adapter.model, "Memory Analysis")
```

## 📊 Understanding the Output

### Model Memory Footprint
- **Total Parameters**: Total number of parameters in the model
- **Trainable Parameters**: Parameters that can be trained (usually same as total)
- **Parameter Memory**: Actual memory used by model weights
- **Estimated Inference Memory**: Memory needed for inference (includes overhead)
- **Estimated Training Memory**: Memory needed for training (includes gradients/optimizer)

### Device Mapping
Shows exactly which layers/modules are placed on which devices:
- `0`, `1`, `2`, etc. = GPU indices
- `cpu` = CPU RAM
- `disk` = Hard disk storage

### GPU Memory Usage
For each GPU:
- **Allocated**: Memory currently in use by tensors
- **Reserved**: Memory reserved by PyTorch (includes allocated + cache)
- **Free**: Available VRAM on the GPU
- **Total**: Total VRAM capacity
- **Percentage Used**: Reserved memory as % of total

## 💡 Practical Examples

### Example 1: Your Custom Device Map

```python
# Your existing Mixtral setup with memory monitoring
config = ModelConfig(
    model_id="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
    device_map={
        'model.embed_tokens': 0,
        'model.layers.0': 0, 'model.layers.1': 0,
        # ... your custom mapping
        'model.layers.31': 1,
        'model.norm': 1, 'lm_head': 1
    },
    load_in_4bit=True,
    show_memory_usage=True  # Will show how your mapping affects memory
)

adapter = HuggingFaceLocalAdapterV2(model_config=config)
```

### Example 2: Memory Monitoring During Generation

```python
from src.hf_local_adapter.utils import print_gpu_memory_usage

# Your existing generation code
def generate_text(prompt, llm_model, tokenizer):
    print_gpu_memory_usage("Before Generation")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    # ... your generation code ...
    
    print_gpu_memory_usage("After Generation")  
    return tokens_per_second

# Use in your existing while loop
while True:
    user_question = input("Please enter your question: ")
    if user_question.lower() == 'exit':
        break
    
    prompt = PHI_PROMPT.format(query=user_question)
    tps = generate_text(prompt, model, tokenizer)
    print(f"Token generation speed: {tps} tokens/second")
```

### Example 3: Debugging Memory Issues

```python
from src.hf_local_adapter.utils import get_gpu_memory_info

# Get detailed memory info programmatically
memory_info = get_gpu_memory_info()

for gpu_id, info in memory_info.items():
    if info['free'] < 2.0:  # Less than 2GB free
        print(f"⚠️  GPU {gpu_id} is running low on memory!")
        print(f"   Free: {info['free']:.2f}GB / {info['total']:.2f}GB")
```

## 🔧 Integration Tips

1. **Keep monitoring enabled during development** to understand memory usage patterns
2. **Disable in production** if you don't want the console output: `show_memory_usage=False`
3. **Use manual monitoring** in generation loops to track memory usage over time
4. **Check device mapping** to ensure your custom device maps are working as expected

## 🚨 Troubleshooting

- **"CUDA not available"**: Normal on CPU-only systems
- **No memory reduction with device mapping**: Check if your layers are being split correctly
- **Unexpected memory usage**: Use `print_model_device_map()` to verify layer placement

This will help optimizing memory usage and debug device mapping issues more effectively. 