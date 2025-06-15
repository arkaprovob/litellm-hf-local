# Quantization Dependencies Troubleshooting Guide

## Issue: BitsAndBites Compilation Errors on Ubuntu

When using the quantization features (`uv sync --extra quantization`) on Ubuntu systems, you may encounter compilation errors during runtime when initializing models with quantization enabled.

### Problem 1: Missing C Compiler

**Error:**
```
RuntimeError: Failed to find C compiler. Please specify via CC environment variable.
```

**Root Cause:**
- `bitsandbytes` requires compilation of CUDA utilities at runtime
- Triton (dependency of bitsandbytes) needs to compile C code on-the-fly
- Ubuntu systems often don't have build tools installed by default

**Solution:**
```bash
sudo apt-get update && sudo apt-get install -y build-essential
```

**What this installs:**
- `gcc` - GNU C compiler
- `g++` - GNU C++ compiler  
- `make` - Build automation tool
- Other essential build tools

### Problem 2: Missing Python Development Headers

**Error:**
```
fatal error: Python.h: No such file or directory
    5 | #include <Python.h>
      |          ^~~~~~~~~~
compilation terminated.
```

**Root Cause:**
- Triton needs Python development headers to compile C extensions
- `Python.h` is required for creating Python C extensions
- Standard Python installation doesn't include development headers

**Solution:**
```bash
sudo apt-get install -y python3.12-dev
```

**What this installs:**
- Python development headers (`Python.h`)
- Static libraries for Python
- Configuration files needed for compiling Python extensions

### Complete Fix for Ubuntu 24.04

```bash
# Install both dependencies in one command
sudo apt-get update && sudo apt-get install -y build-essential python3.12-dev
```

## Why This Happens

### The Quantization Stack
1. **Your Code** → Uses `load_in_4bit=True`
2. **Transformers** → Calls quantization backend
3. **BitsAndByes** → Provides 4-bit/8-bit quantization
4. **Triton** → GPU kernel compilation framework
5. **CUDA Utils** → Need C compilation at runtime

### Runtime Compilation
Unlike pre-compiled packages, quantization libraries often:
- Compile optimized kernels on first use
- Generate hardware-specific code
- Require build tools to be available at runtime

## Docker Configuration

When building Docker images with quantization support:

```dockerfile
# Ubuntu-based Dockerfile
FROM ubuntu:24.04

# Install system dependencies for quantization
RUN apt-get update && apt-get install -y \
    build-essential \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY . /app
WORKDIR /app

# Install with quantization support
RUN uv sync --extra quantization

# Your application setup...
```

## System Requirements Summary

### For Development (Ubuntu 24.04)
```bash
# Required for quantization features
sudo apt-get install -y build-essential python3.12-dev

# Optional but recommended
sudo apt-get install -y \
    git \
    curl \
    nvidia-cuda-toolkit  # If using CUDA
```

### For Production Deployment
- Same build dependencies required
- CUDA toolkit if using GPU quantization
- Sufficient GPU memory for quantized models

## Alternative Solutions

### 1. CPU-only Mode
If you don't need GPU quantization:
```python
# Disable quantization for CPU-only usage
adapter = HuggingFaceLocalAdapterV2(
    model_id="microsoft/Phi-4-reasoning",
    device="cpu",  # Force CPU
    load_in_4bit=False,  # Disable quantization
)
```

### 2. Pre-compiled Wheels
Some platforms offer pre-compiled wheels that don't require compilation:
```bash
# Check if pre-compiled wheels are available
uv pip install --only-binary=all bitsandbytes
```

### 3. Alternative Quantization
Consider other quantization methods that don't require runtime compilation:
```python
# Use torch's native quantization instead
adapter = HuggingFaceLocalAdapterV2(
    model_id="microsoft/Phi-4-reasoning",
    torch_dtype=torch.float16,  # Half precision instead of 4-bit
)
```

## Testing Your Setup

After installing the dependencies, verify everything works:

```bash
# Test C compiler
gcc --version

# Test Python headers
python3.12-config --includes

# Test quantization import
uv run python -c "import bitsandbytes; print('✅ BitsAndByes working')"

# Test your adapter
uv run python tests/test_lightllm.py
```

## Platform-Specific Notes

### Ubuntu/Debian
```bash
sudo apt-get install -y build-essential python3.12-dev
```

### CentOS/RHEL/Fedora
```bash
sudo yum groupinstall "Development Tools"
sudo yum install python3.12-devel
```

### Alpine Linux (Docker)
```bash
apk add --no-cache build-base python3-dev
```

### macOS
```bash
# Usually works out of the box with Xcode Command Line Tools
xcode-select --install
```

This issue is specific to systems where Python packages need to compile C extensions at runtime. The quantization libraries are performance-critical and often compile optimized code for your specific hardware configuration.