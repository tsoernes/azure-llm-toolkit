# Contributing to Azure LLM Toolkit

Thank you for your interest in contributing to Azure LLM Toolkit! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to foster an inclusive and welcoming community.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- An Azure OpenAI account (for testing)

### Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/azure-llm-toolkit.git
cd azure-llm-toolkit
```

2. **Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -e ".[dev]"
```

4. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

5. **Run tests to verify setup**

```bash
pytest
```

## Development Workflow

### Branch Naming

- `feature/` - New features (e.g., `feature/add-streaming-support`)
- `fix/` - Bug fixes (e.g., `fix/rate-limiter-edge-case`)
- `docs/` - Documentation changes (e.g., `docs/improve-readme`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-config`)

### Making Changes

1. **Create a feature branch**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**

   - Write clear, concise code
   - Follow the existing code style
   - Add docstrings to new functions/classes
   - Update type hints

3. **Add tests**

   - Write tests for new functionality
   - Ensure existing tests still pass
   - Aim for high test coverage

4. **Update documentation**

   - Update README.md if adding user-facing features
   - Add docstrings with examples
   - Update CHANGELOG.md

5. **Run quality checks**

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
basedpyright src/
mypy src/

# Run tests
pytest

# Run tests with coverage
pytest --cov=azure_llm_toolkit --cov-report=html
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 120 characters (configured in pyproject.toml)
- **Imports**: Organized using `ruff` (isort rules)
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Google-style docstrings for all public APIs

### Type Hints

Use modern Python type hints:

```python
# Good
def process_items(items: list[str]) -> dict[str, int]:
    ...

# Avoid
from typing import List, Dict
def process_items(items: List[str]) -> Dict[str, int]:
    ...
```

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.

    Longer description if needed. Can span multiple lines.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When input is invalid
        RuntimeError: When operation fails

    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        True
    """
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_client.py

# Run with coverage
pytest --cov=azure_llm_toolkit --cov-report=html

# Run tests matching a pattern
pytest -k "test_embeddings"
```

### Writing Tests

- Use `pytest` conventions
- Use `pytest-asyncio` for async tests
- Mock external API calls when possible
- Test both success and failure cases

Example test:

```python
import pytest
from azure_llm_toolkit import AzureConfig, AzureLLMClient

@pytest.mark.asyncio
async def test_embed_texts():
    """Test basic text embedding functionality."""
    config = AzureConfig()
    client = AzureLLMClient(config=config)
    
    texts = ["test text"]
    result = await client.embed_texts(texts)
    
    assert len(result.embeddings) == 1
    assert len(result.embeddings[0]) > 0
    assert result.usage.total_tokens > 0
```

## Pull Request Process

1. **Ensure all checks pass**

   - All tests pass
   - Code is formatted
   - No linting errors
   - Type checking passes

2. **Update documentation**

   - Update README.md if needed
   - Add entry to CHANGELOG.md
   - Update docstrings

3. **Create pull request**

   - Use a clear, descriptive title
   - Reference related issues (e.g., "Fixes #123")
   - Provide detailed description of changes
   - Include examples if adding new features

4. **PR Template**

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] All tests pass
- [ ] Code formatted with ruff
- [ ] Type checking passes

## Related Issues
Fixes #(issue number)
```

## Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add support for streaming responses

- Implement streaming for chat completions
- Add tests for streaming functionality
- Update documentation with streaming examples"

# Less good
git commit -m "fix bug"
```

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## Reporting Issues

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (Python version, OS, etc.)
- Code samples if applicable

### Feature Requests

Include:
- Clear description of the feature
- Use cases and motivation
- Example API or usage pattern
- Any alternatives considered

## Documentation

### Updating Documentation

- Keep README.md up to date
- Add examples for new features
- Update API reference if needed
- Check for broken links

### Documentation Style

- Use clear, concise language
- Provide code examples
- Include expected output
- Link to related concepts

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag: `git tag -a v0.2.0 -m "Release v0.2.0"`
4. Push tag: `git push origin v0.2.0`
5. Build and publish to PyPI:

```bash
# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

## Questions?

If you have questions about contributing:

- Open an issue with the `question` label
- Check existing issues and discussions
- Review the README.md and documentation

Thank you for contributing to Azure LLM Toolkit! 🎉