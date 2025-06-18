# Contributing to litellm-hf-local

Thank you for your interest in contributing to the litellm-hf-local project! This guide will help you understand how to contribute effectively and follow the project's standards.

## Table of Contents

1. [Code Style Guidelines](#code-style-guidelines)
2. [Submitting Changes](#submitting-changes)
3. [Development Setup](#development-setup)
4. [Testing](#testing)
5. [Documentation](#documentation)

## Code Style Guidelines

### Type Annotations

- Always use explicit type annotations for all function parameters and return values
- Prefer concrete types over Any when possible
- Use Union[Type1, Type2] for multiple possible types (or | operator in Python 3.10+)
- Use Optional[Type] for parameters that could be None

```python
# Good
def process_response(response: Dict[str, Any], max_tokens: int = 100) -> List[str]:
    """Process the model response and return a list of tokens."""
    # Implementation

# Avoid
def process_response(response, max_tokens=100):
    # Implementation
```

### Code Documentation

- Add class-level docstrings explaining the purpose and functionality of each class
- Add method-level docstrings for all methods, especially public ones
- Format docstrings using standard Python docstring conventions

```python
class ModelAdapter:
    """
    A base adapter class for interacting with LLM models.
    
    This class provides the interface and common functionality for 
    all model adapters in the system.
    """
    
    def generate_text(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Generate text based on the provided prompt and parameters.
        
        Args:
            prompt: The input prompt to generate from
            params: Model configuration parameters
            
        Returns:
            The generated text response
        """
        # Implementation
```

### Code Structure

- Avoid deeply nested conditionals and loops
- Keep nesting depth to a maximum of 2 levels
- Extract complex logic into separate, well-named functions
- Keep functions focused on a single responsibility

```python
# Good
def process_config(config: ModelConfig) -> Dict[str, Any]:
    """Process the model configuration."""
    validated_config = validate_config(config)
    return prepare_for_model(validated_config)
    
def validate_config(config: ModelConfig) -> ModelConfig:
    """Validate the model configuration parameters."""
    # Validation logic
    return config
    
def prepare_for_model(config: ModelConfig) -> Dict[str, Any]:
    """Convert config to model-ready format."""
    # Conversion logic
    return model_config

# Avoid
def process_config(config: ModelConfig) -> Dict[str, Any]:
    """Process the model configuration."""
    if config.model_id:
        if config.device == "cuda":
            # Deep nesting - hard to follow
            for param in config.parameters:
                if param.value > 0:
                    # More nesting...
    # More code...
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the style guidelines
4. Add or update tests as needed
5. Update documentation to reflect your changes
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

1. Clone the repository
2. Install dependencies:
   ```
   uv venv .venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

## Testing

- Write tests for all new features and bug fixes
- Run the test suite before submitting changes
- Ensure existing tests continue to pass
- Use parameterized tests when testing similar functionality with different inputs

```
python -m pytest tests/
```

## Documentation

- Update or add documentation for all changes
- Document new features, configuration options, and API changes
- Provide examples for new functionality where appropriate

## Code Refactoring

If you see an anti-pattern or shortcut in the codebase and have a better solution:

1. Create an issue describing the problem and your proposed solution
2. Discuss the approach with maintainers before implementing major changes
3. Provide a clear justification for your approach
4. Ensure backward compatibility where possible
5. Add proper tests for your new implementation

Remember that good code is written for humans to read and understand. When in doubt, choose clarity over cleverness.

Thank you for helping improve the litellm-hf-local project!
