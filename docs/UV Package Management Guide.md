# LiteLLM HF Local - UV Package Management Guide

A brief guide for managing the LiteLLM HF Local package using uv, covering installation, dependency management,
and development workflows.

## Requirements

- Python 3.12 or higher
- CUDA-compatible GPU (recommended for best performance)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### 1. Install uv (if not already installed)

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip (if you must)
pip install uv
```

### 2. Basic Installation

```bash
# Clone the repository
git clone https://github.com/arkaprovob/llm-hf-local-adapter
cd llm-hf-local-adapter

# Install core dependencies only
uv sync
```

### 3. Installation with Optional Features

Install additional features based on your needs:

```bash
# Install with 4-bit/8-bit quantization support
uv sync --extra quantization

# Install with advanced tokenizer support (for Google/Meta models)
uv sync --extra tokenizer

# Install all optional dependencies (recommended)
uv sync --all-extras
```

### 4. Development Setup

For contributors and developers:

```bash
# Clone and setup in one go
git clone https://github.com/arkaprovob/llm-hf-local-adapter
cd llm-hf-local-adapter

# uv automatically creates .venv and installs dependencies
uv sync --all-extras
```

### 5. Quick Start (One Command)

```bash
# Run directly with uv (no installation needed)
uv run --extra all your_script.py
```

## Package Dependencies

### Core Dependencies (Always Installed)

- **accelerate** (≥1.7.0) - Distributed training and inference
- **litellm** (≥1.72.6) - Unified LLM API interface
- **torch** (≥2.7.1) - PyTorch deep learning framework
- **transformers** (≥4.52.4) - HuggingFace model library

### Optional Dependencies

#### Quantization Support (`[quantization]`)

- **bitsandbytes** (≥0.46.0) - 4-bit and 8-bit model quantization

**When to use:** Running large models (7B+ parameters) on limited GPU memory

```bash
uv sync --extra quantization
```

#### Advanced Tokenizer Support (`[tokenizer]`)

- **protobuf** (≥6.31.1) - Protocol buffer support
- **sentencepiece** (≥0.2.0) - Advanced tokenization for certain models

**When to use:** Working with models that require SentencePiece tokenization (LLaMA, T5, etc.)

```bash
uv sync --extra tokenizer
```

#### All Features (`[all]`)

Includes all optional dependencies for full functionality.

```bash
uv sync --all-extras
```

## Building the Package

### For Distribution

```bash
# Build wheel and source distribution
uv build

# This creates:
# - dist/litellm_hf_local-0.1.0-py3-none-any.whl
# - dist/litellm_hf_local-0.1.0.tar.gz
```

### For Development

```bash
# Install in editable mode (changes reflect immediately)
uv sync

# With specific features
uv sync --extra quantization

# Or update dependencies after editing pyproject.toml
uv lock  # Update lockfile only
uv sync  # Update lockfile and environment
```

### Quick Start Examples

## Managing Dependencies

### Adding New Dependencies

```bash
# Add a core dependency
uv add torch>=2.8.0

# Add an optional dependency to a specific group
uv add --optional quantization bitsandbytes>=0.46.0

# Add a development dependency  
uv add --dev pytest

# Add to custom dependency group
uv add --group quantization your-new-package
```

### Legacy pip-style Installation (Alternative)

If you need to use pip-style commands for compatibility:

```bash
# Install with core dependencies only
uv pip install -e .

# Install with specific optional dependencies
uv pip install -e ".[quantization]"
uv pip install -e ".[tokenizer]"
uv pip install -e ".[all]"

# Install multiple optional groups
uv pip install -e ".[quantization,tokenizer]"
```

### Updating Dependencies

```bash
# Update all dependencies
uv lock --upgrade

# Update a specific package
uv lock --upgrade-package torch

# Sync after updating
uv sync
```

### Working with Lock Files

```bash
# Generate/update lockfile without installing
uv lock

# Install from existing lockfile (exact versions)
uv sync --frozen

# Install ignoring lockfile (latest compatible versions)
uv sync --no-lock

# Update specific packages in lockfile
uv lock --upgrade-package torch
uv lock --upgrade-package bitsandbytes

# Force refresh of entire lockfile
uv lock --upgrade
```

### Advanced Dependency Management

```bash
# Remove dependencies
uv remove torch
uv remove --optional quantization bitsandbytes
uv remove --dev pytest

# Sync without development dependencies
uv sync --no-dev

# Sync only specific groups
uv sync --only-group quantization
uv sync --group quantization --group tokenizer

# Exclude specific groups
uv sync --no-group dev

# Install without build isolation (for complex packages)
uv sync --no-build-isolation
```

Test your installation:

```bash
# Check if all dependencies are installed
uv tree

# Test basic functionality
uv run python -c "import litellm; from hf_local_adapter import HuggingFaceLocalAdapterV2; print('✅ Installation successful!')"

# Test quantization support (if installed)
uv run python -c "import bitsandbytes; print('✅ Quantization support available')"
```

## Troubleshooting

### CUDA Issues

```bash
# Check CUDA availability
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Memory Issues

- Use quantization: Install with `uv sync --extra quantization`
- Reduce model size or use smaller models
- Clear GPU cache: `torch.cuda.empty_cache()`

## Common UV Patterns & Tips

### Project Structure Best Practices

```
your-project/
├── .venv/              # Virtual environment (auto-created)
├── src/                # Source code
├── tests/              # Test files
├── pyproject.toml      # Project configuration
├── uv.lock            # Locked dependencies (commit this!)
├── .python-version    # Python version (optional)
└── README.md
```

### Useful UV Aliases

```bash
# Add to your shell profile
alias uvs="uv sync"
alias uvr="uv run"
alias uva="uv add"
alias uvb="uv build"
alias uvt="uv tree"
```

### Environment Variables

```bash
# Customize UV behavior
export UV_PROJECT_ENVIRONMENT=".venv"  # Custom venv location
export UV_CACHE_DIR="~/.cache/uv"      # Cache directory
export UV_NO_SYNC=1                    # Disable auto-sync

# For CI/CD
export UV_SYSTEM_PYTHON=1              # Use system Python
```

### Troubleshooting Commands

```bash
# Clear UV cache
uv cache clean

# Reinstall everything
uv sync --reinstall

# Verbose output for debugging
uv sync --verbose

# Check UV version and config
uv --version
uv config list
```

## Package Information

- **Package Name**: `litellm-hf-local`
- **Version**: 0.1.0
- **Python**: 3.12+
- **License**: See LICENSE file
- **Homepage**: https://github.com/arkaprovob/llm-hf-local-adapter

## Contributing & Development

### Setting Up Development Environment

```bash
# Clone and setup in one go
git clone https://github.com/arkaprovob/llm-hf-local-adapter
cd llm-hf-local-adapter

# Install all dependencies including optional ones
uv sync --all-extras
```

### Adding Dependencies During Development

```bash
# Add core dependencies
uv add transformers>=4.52.4

# Add optional dependencies to specific groups
uv add --optional quantization bitsandbytes>=0.46.0
uv add --optional tokenizer sentencepiece>=0.2.0

# Add to custom dependency groups
uv add --group quantization auto-gptq>=0.4.0
uv add --group inference vllm>=0.2.0

# Add development dependencies
uv add --dev pytest black ruff mypy
```

### Editing Dependencies Manually

You can also edit `pyproject.toml` directly and then sync:

```toml
[project.optional-dependencies]
quantization = [
    "bitsandbytes>=0.46.0",
    "auto-gptq>=0.4.0",  # Added manually
]
```

```bash
# After editing pyproject.toml
uv lock  # Update lockfile
uv sync  # Install new dependencies
```

### Development Workflow

1. Make your changes
2. Test your changes:
   ```bash
   uv run python -m pytest tests/
   ```
3. Build and verify:
   ```bash
   uv build
   ```
4. Update dependencies if needed:
   ```bash
   uv lock --upgrade
   uv sync
   ```

## Support

- 📖 **Documentation**: [GitHub README](https://github.com/arkaprovob/litellm-hf-local/blob/master/README.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/arkaprovob/litellm-hf-local/issues)
- 💬 **Discussions**: Create an issue for questions

---

**Note**: This adapter requires CUDA-compatible hardware for optimal performance. CPU inference is supported but will be
significantly slower.