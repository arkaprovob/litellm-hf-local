# 🤗 LiteLLM HuggingFace Local Adapter

> **Finally! Out-of-the-box HuggingFace model support for LiteLLM with bitsandbytes quantization** 

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🚀 Why This Exists

**TL;DR: For the experimenters, the tinkerers, and the "I want it now" crowd.**

After waiting **two years** for someone to add proper HuggingFace offline model support to LiteLLM (or similar tools), I decided to build it myself. While the world went full Ollama mode, some of us remained loyal to the HuggingFace ecosystem ..you know, the ones who:

- 🧪 **Experiment constantly** with the latest models
- 🎯 **Want bleeding-edge models** without waiting for GGUF conversions  
- 💾 **Love bitsandbytes quantization** (4-bit/8-bit) on the go
- 🔥 **Need it to work out-of-the-box** with clean, maintainable code
- ⚡ **Want solid features** writing same block of codes and functions repeatedly

This adapter aims to bridge that gap with a focus on clean architecture, useful chat template handling, and the features you actually need. It's still evolving, and there's definitely room for improvement – that's where the community comes in! 🤝

### 🔗 **Why LiteLLM Matters**

LiteLLM has become the de facto abstraction layer for LLM integrations across the ecosystem. Instead of hardcoding OpenAI, Gemini, or Claude APIs, modern frameworks use LiteLLM for provider flexibility:

- **🚀 Google ADK**: Uses LiteLLM for model-agnostic agent development
- **🛠️ LangChain**: Integrates with LiteLLM for unified model access
- **⚡ CrewAI**: Leverages LiteLLM for multi-model support
- **🏗️ AutoGen**: Supports LiteLLM for diverse model backends

**The Problem**: These frameworks work great with cloud APIs, but what if you want to use your locally hosted HuggingFace models?

**The Solution**: This adapter makes your local HuggingFace models available through LiteLLM's interface, so they work seamlessly with any LiteLLM-compatible framework.

```python
# Now your local models work with any LiteLLM-based framework!
import litellm
from your_favorite_framework import Agent

# Register your local model
litellm.custom_provider_map = [
    {"provider": "huggingface-local", "custom_handler": your_adapter}
]

# Use it in any framework that supports LiteLLM
agent = LiteLlm(model="huggingface-local/your-model")
```

## ✨ Key Features

### 🎯 **Out-of-the-Box Excellence**
- **LiteLLM Integration**: Works great with LiteLLM's completion interface
- **Chat Template Support**: Includes templates for popular model families (Llama-2, Mistral, Falcon, ChatML, etc.) - experimental feature with fallbacks
- **Smart Tokenization**: Proper token counting with context window validation
- **Memory Management**: Device mapping for multi-GPU setups (defaults work for most cases, but MOE models may need custom device_map)

### 🧠 **Advanced Quantization**
- **4-bit & 8-bit Quantization**: Built-in bitsandbytes support for memory efficiency
- **Quantization On-the-Go**: Fine-tune 4-bit/8-bit inferencing as needed for your specific use case, with the same weights
- **Memory Monitoring**: Built-in GPU memory usage tracking and reporting

### ⚡ **HuggingFace Features Made Easy**
- **Streaming Support**: Leverages HuggingFace's TextIteratorStreamer with proper LiteLLM chunk formatting (TBH proud of this one; took some digging to figure out!)
- **Async Operations**: Async/await support for the completion interface
- **Token Counting**: Integrates HuggingFace tokenizer for accurate token tracking
- **Generation Control**: Exposes HuggingFace generation parameters with sensible defaults (advanced users can override)

### 🏗️ **Clean & Hackable Architecture**
- **Modular Design**: Separate components for config, formatting, generation, and utilities
- **Type Hints**: Type annotations to help you understand the codebase
- **Room for Growth**: Test suite exists, but there's always room for more coverage (I am drowning in deadlines, so let's write more tests together!)
- **Contributor-Friendly**: Designed to be easy to understand, extend, and improve

## 📦 Installation

### Basic Installation
```bash
# Install with uv (recommended)
uv add litellm-hf-local

# Or with pip
pip install litellm-hf-local
```

### With Quantization Support
```bash
# Install with quantization dependencies
uv sync --extra quantization

# Or with pip
pip install "litellm-hf-local[quantization]"
```

### All Features
```bash
# Install everything
uv sync --extra all

# Or with pip  
pip install "litellm-hf-local[all]"
```

> **💡 Ubuntu Users**: If you encounter compilation errors, see our [Quantization Dependencies Troubleshooting Guide](docs/Quantization%20Dependencies%20Troubleshooting%20Guide.md)

## 🚀 Quick Start

### Basic Usage

```python
from src import HuggingFaceLocalAdapterV2, ModelConfig
import litellm

# Configure your model
config = ModelConfig(
    model_id="microsoft/Phi-4-reasoning",
    device="cuda:0",
    load_in_4bit=True,  # Enable 4-bit quantization
    trust_remote_code=True
)

# Create the adapter
adapter = HuggingFaceLocalAdapterV2(
    model_config=config,
    context_window=4096,
    temperature=0.8,
    max_new_tokens=512
)

# Register with LiteLLM
litellm.custom_provider_map = [
    {"provider": "huggingface-local", "custom_handler": adapter}
]

# Use it like any LiteLLM model!
response = litellm.completion(
    model="huggingface-local/Phi-4-reasoning",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
)

print(response.choices[0].message.content)
```

### Streaming Example

```python
# Enable streaming
response = litellm.completion(
    model="huggingface-local/Phi-4-reasoning", 
    messages=[
        {"role": "user", "content": "Write a story about AI and humanity."}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Advanced Configuration

```python
from src import HuggingFaceLocalAdapterV2, ModelConfig
import torch

# Advanced model configuration
config = ModelConfig(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    device="cuda:0",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "10GB", 1: "10GB"},
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }
)

# Create adapter with custom generation parameters
adapter = HuggingFaceLocalAdapterV2(
    model_config=config,
    context_window=8192,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    stopping_ids=[128001, 128009],  # Custom stop tokens
)
```

## 🎯 Model Compatibility

The adapter should work with **most HuggingFace text-to-text causal language models**. The chat template detection tries to handle popular model families, but there's always room for improvement.

### ✅ **Tested Models**
- `Qwen/Qwen2.5-7B-Instruct` - Works great
- `microsoft/Phi-4-reasoning` - Tested extensively
- `google/gemma-3-27b-it` - Works great

### 🤔 **Should Work (But Not Thoroughly Tested)**
The chat template system has fallbacks, so even if auto-detection fails, you'll still get output. You can provide custom functions  for the chat template if needed.

### 🚫 **Won't Work**
- Models requiring special preprocessing
- Non-causal language models
- Models with custom architectures that transformers doesn't support; I will open an issue for these cases and feel free to add more in the [issues](https://github.com/arkaprovob/litellm-hf-local/issues) if something comes to mind!

> **💡 Tip**: New models should work immediately since this uses standard HuggingFace transformers!

## 🏗️ Architecture (this might be a bit outdated)

```
litellm-hf-local/
├── src/hf_local_adapter/
│   ├── adapter.py              # Main adapter class
│   ├── config/
│   │   └── model_config.py     # Model configuration
│   ├── formatting/
│   │   ├── message_formatter.py # Chat template handling
│   │   └── templates.py        # Template definitions
│   ├── generation/
│   │   ├── parameters.py       # Generation parameters
│   │   └── stopping_criteria.py # Advanced stopping
│   └── utils/
│       ├── tokenization.py     # Token utilities
│       └── logging.py          # Logging setup
```

### 🧩 **Design Principles**

1. **Separation of Concerns**: Each component has a single responsibility
2. **Type Safety**: Comprehensive type annotations throughout
3. **Extensibility**: Easy to add new models and features  
4. **Performance**: Optimized for both memory and speed
5. **Maintainability**: Clean, documented, testable code

## 🔧 Configuration Options

### ModelConfig Parameters

```python
@dataclass
class ModelConfig:
    model_id: str                          # HuggingFace model identifier
    device: str = "cuda:0"                 # Device to load model on
    cache_dir: Optional[str] = None        # Model cache directory
    trust_remote_code: bool = False        # Trust remote code
    load_in_4bit: bool = False            # Enable 4-bit quantization
    load_in_8bit: bool = False            # Enable 8-bit quantization
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: Union[str, Dict] = "auto"  # Device mapping strategy  
    max_memory: Optional[Dict] = None      # Memory limits per device
    quantization_config: Optional[Dict] = None  # Custom quantization
    show_memory_usage: bool = True         # Show GPU memory usage after loading
```

### Generation Parameters

```python
# Available in HuggingFaceLocalAdapterV2 constructor
temperature: float = 1.0           # Sampling temperature
top_p: float = 1.0                 # Nucleus sampling
top_k: int = 50                    # Top-k sampling  
repetition_penalty: float = 1.0    # Repetition penalty
do_sample: bool = True             # Enable sampling
max_new_tokens: int = 512          # Maximum tokens to generate
context_window: int = 4096         # Model context window
```

## 📊 Performance & Memory

### Memory Usage Examples

| Model | Precision | Memory (GB) | 4-bit Quantized |
|-------|-----------|-------------|-----------------|
| Llama-3.1-8B | bfloat16 | ~16 GB | ~4.5 GB |
| Phi-4 | bfloat16 | ~28 GB | ~8 GB |  
| Mistral-7B | bfloat16 | ~14 GB | ~4 GB |

### Quantization Benefits

- **4-bit Quantization**: ~75% memory reduction with minimal quality loss
- **8-bit Quantization**: ~50% memory reduction with negligible quality loss
- **Memory Awareness**: Built-in memory monitoring and reporting
- **Multi-GPU Support**: Distribute large models across multiple GPUs

### 🔍 Memory Monitoring

The adapter automatically displays detailed memory information when models are loaded:

```python
from src import HuggingFaceLocalAdapterV2, ModelConfig

# Memory monitoring enabled by default
config = ModelConfig(
    model_id="microsoft/Phi-4-reasoning",
    device_map="auto",
    max_memory={0: "10GB", 1: "10GB"},
    show_memory_usage=True  # Default: True
)

adapter = HuggingFaceLocalAdapterV2(model_config=config)
```

**Output:**
```
=== Model Memory Footprint: microsoft/Phi-4-reasoning ===
Total Parameters: 14,701,875,200
Trainable Parameters: 14,701,875,200
Parameter Memory: 27.35 GB
Estimated Inference Memory: 32.82 GB

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
GPU 0 (NVIDIA GeForce RTX 3090): 9.23GB allocated, 9.87GB reserved, 14.13GB free / 24.00GB total (41.1% used)
GPU 1 (NVIDIA GeForce RTX 3090): 8.91GB allocated, 9.45GB reserved, 14.55GB free / 24.00GB total (39.4% used)
```

**Manual Memory Monitoring:**
```python
from src.hf_local_adapter.utils import (
    print_gpu_memory_usage,
    print_model_device_map,
    print_model_memory_footprint
)

# Call anytime during execution
print_gpu_memory_usage("After Generation")
print_model_device_map(model, "Current Device Map")
```

**Disable Memory Monitoring:**
```python
config = ModelConfig(
    model_id="microsoft/Phi-4-reasoning",
    show_memory_usage=False  # Disable automatic monitoring
)
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
# Run all tests
uv run python -m pytest tests/

# Run specific test
uv run python tests/test_lightllm.py

# Test with quantization
uv run python tests/test_quantization.py
```

## 🚧 Roadmap

### 🎯 **Coming Soon**
- [ ] **Benchmark Suite**: Comprehensive performance benchmarking
- [ ] **Vision Models**: Support for multimodal models
- [ ] **Custom Samplers**: Advanced sampling strategies
- [ ] **Model Caching**: Intelligent model caching system
- [ ] **Batch Processing**: Efficient batch inference

### 💡 **Ideas Welcome**
- What models do you want to see supported?
- What features would make your workflow easier?
- Found a bug or have an improvement idea?

## 🤝 Contributing

Contributions are welcome! This project was built for the community, and it's meant to grow with community input.

### 🔥 **Ways to Contribute**
- **Test new models** and report compatibility
- **Add chat templates** for unsupported model families  
- **Improve performance** with optimizations
- **Fix bugs** and improve reliability
- **Add features** that benefit the community

See our [Contributing Guide](CONTRIBUTING.md) for detailed instructions.

### 💬 **Join the Discussion**
- 🐛 **Found a bug?** [Open an issue](https://github.com/arkaprovob/litellm-hf-local/issues)
- 💡 **Have an idea?** [Start a discussion](https://github.com/arkaprovob/litellm-hf-local/discussions)
- 🚀 **Want to contribute?** Check out [good first issues](https://github.com/arkaprovob/litellm-hf-local/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

## ⚡ Why Choose This Over Ollama?

| Feature | litellm-hf-local | Ollama                     |
|---------|------------------|----------------------------|
| **Latest Models** | ✅ Immediate access | ⏳ Wait for GGUF conversion |
| **Quantization** | ✅ bitsandbytes (4/8-bit) | ✅ GGUF quantization        |  
| **HuggingFace Native** | ✅ Direct integration | ❌ Requires conversion      |
| **Streaming** | ✅ Full streaming support | ✅ Yes                      |
| **Multi-GPU** | ✅ Advanced device mapping | ⚠️ IDK                     |

> **Choose Ollama if**: You want simplicity and don't mind waiting for model support  
> **Choose litellm-hf-local if**: You want the latest models NOW with maximum flexibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LiteLLM Team**: For creating an amazing unified LLM interface
- **HuggingFace**: For the incredible transformers library and model ecosystem  
- **bitsandbytes**: For making quantization accessible and efficient
- **The Community**: For feedback, testing, and contributions

---

<div align="center">

**Built with ❤️ for the HuggingFace community**

*"Because waiting two years for a feature is two years too long"*

[⭐ Star us on GitHub](https://github.com/arkaprovob/litellm-hf-local) • [📖 Documentation](https://github.com/arkaprovob/litellm-hf-local#readme) • [🐛 Report Bug](https://github.com/arkaprovob/litellm-hf-local/issues) • [💡 Request Feature](https://github.com/arkaprovob/litellm-hf-local/issues)

</div>
