# LiteLLM Proxy Setup Guide

This guide explains how to use the HuggingFace Local Adapter with LiteLLM's proxy server to serve local HuggingFace models through an OpenAI-compatible API.

## Overview

The LiteLLM proxy server allows you to serve your local HuggingFace models with an OpenAI-compatible REST API. This enables you to use any OpenAI SDK or tool with your local models.

## Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended)
- Installed dependencies from `pyproject.toml`

## Setup Process

### 1. Configure Your Model

Edit the `custom_handler.py` file to configure your desired HuggingFace model:

```python
from src import ModelConfig, HuggingFaceLocalAdapterV2

config = ModelConfig(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",  # Change to your model
    device="cuda:0",                         # Or "cpu" for CPU inference
    load_in_4bit=False,                      # Set to True for 4-bit quantization
    trust_remote_code=False
)

# Create adapter with your preferred settings
adapter = HuggingFaceLocalAdapterV2(
    model_config=config,
    context_window=4096,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    do_sample=True,
    max_new_tokens=512,
)
```

### 2. Configure LiteLLM Proxy

The `litellm_config.yaml` file defines how the proxy server exposes your model:

```yaml
model_list:
  - model_name: qwen-local                        # Name clients will use
    litellm_params:
      model: huggingface-local/Qwen2.5-0.5B-Instruct  # Provider/model format

# Proxy settings
litellm_settings:
  custom_provider_map:
  - {"provider": "huggingface-local", "custom_handler": custom_handler.adapter}
```

### 3. Start the Server

Activate your virtual environment and start the LiteLLM proxy:

```bash
# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows

# Start the proxy server
litellm --config litellm_config.yaml
```

The server will start on `http://0.0.0.0:4000` by default.

## API Endpoints

### Health Check

Check if the model is loaded and healthy:

```bash
curl -X GET http://0.0.0.0:4000/health
```

**Expected Response:**
```json
{
    "healthy_endpoints": [
        {
            "model": "huggingface-local/Qwen2.5-0.5B-Instruct",
            "cache": {
                "no-cache": true
            }
        }
    ],
    "unhealthy_endpoints": [],
    "healthy_count": 1,
    "unhealthy_count": 0
}
```

### Chat Completions (Non-Streaming)

Send a chat completion request:

```bash
curl -X POST http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-local",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello! Can you explain what you are?"
        }
    ],
    "max_tokens": 100,
    "temperature": 0.7
}'
```

**Expected Response:**
```json
{
    "id": "chatcmpl-1234567890",
    "created": 1234567890,
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "object": "chat.completion",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Hello! I'm Qwen, an AI assistant created by Alibaba Cloud...",
                "role": "assistant"
            }
        }
    ],
    "usage": {
        "completion_tokens": 45,
        "prompt_tokens": 28,
        "total_tokens": 73
    }
}
```

### Chat Completions (Streaming)

For streaming responses, add `"stream": true`:

```bash
curl -X POST http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-local",
    "messages": [
        {
            "role": "user",
            "content": "Count from 1 to 5 slowly"
        }
    ],
    "max_tokens": 50,
    "stream": true
}'
```

**Expected Response (Server-Sent Events):**
```
data: {"id":"chatcmpl-123","created":1234567890,"model":"Qwen2.5-0.5B-Instruct","choices":[{"index":0,"delta":{"content":"1","role":"assistant"}}]}

data: {"id":"chatcmpl-123","created":1234567890,"model":"Qwen2.5-0.5B-Instruct","choices":[{"index":0,"delta":{"content":"..."}}]}

data: {"id":"chatcmpl-123","created":1234567890,"model":"Qwen2.5-0.5B-Instruct","choices":[{"index":0,"delta":{"content":"5"}}]}

data: [DONE]
```

## Using with OpenAI SDK

You can use the OpenAI Python SDK with your local model:

```python
from openai import OpenAI

# Point to your local server
client = OpenAI(
    base_url="http://0.0.0.0:4000/v1",
    api_key="dummy-key"  # LiteLLM doesn't require auth by default
)

# Non-streaming
response = client.chat.completions.create(
    model="qwen-local",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen-local",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Configuration Options

### Model Configuration

In `custom_handler.py`, you can configure:

- **model_id**: HuggingFace model identifier
- **device**: CUDA device or "cpu"
- **load_in_4bit/load_in_8bit**: Quantization options
- **context_window**: Maximum context length
- **temperature**: Sampling temperature (0.0-2.0)
- **top_p**: Nucleus sampling parameter
- **top_k**: Top-k sampling parameter
- **max_new_tokens**: Maximum tokens to generate

### Proxy Configuration

In `litellm_config.yaml`, you can:

- Define multiple models with different configurations
- Set up load balancing between models
- Configure authentication and rate limiting
- Add custom middleware

## Troubleshooting

### Common Issues

1. **"model_kwargs not recognized" error**: The adapter now filters out LiteLLM-specific parameters. Check that the parameter mapping is working correctly.

2. **Out of memory**: Try enabling 4-bit quantization in `custom_handler.py`:
   ```python
   config = ModelConfig(
       model_id="your-model",
       device="cuda:0",
       load_in_4bit=True
   )
   ```

3. **Slow responses**: 
   - Ensure you're using GPU (`device="cuda:0"`)
   - Consider using a smaller model
   - Reduce `max_new_tokens`

### Debug Mode

For more detailed logging, set the environment variable:
```bash
export LITELLM_LOG=DEBUG
litellm --config litellm_config.yaml
```

## Advanced Usage

### Multiple Models

You can serve multiple models by adding them to `litellm_config.yaml`:

```yaml
model_list:
  - model_name: qwen-small
    litellm_params:
      model: huggingface-local/Qwen2.5-0.5B-Instruct
      
  - model_name: qwen-large  
    litellm_params:
      model: huggingface-local/Qwen2.5-1.5B-Instruct
```

### Custom System Prompts

Configure default system prompts in the adapter initialization:

```python
adapter = HuggingFaceLocalAdapterV2(
    model_config=config,
    default_system_prompt="You are a helpful coding assistant.",
    # ... other parameters
)
```

## References

- [LiteLLM Custom LLM Server Documentation](https://docs.litellm.ai/docs/providers/custom_llm_server)
- [LiteLLM Proxy Server Documentation](https://docs.litellm.ai/docs/proxy/quick_start)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) 