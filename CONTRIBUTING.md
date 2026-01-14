# Contributing to ML-TSSP HUMINT Dashboard

Thank you for your interest in contributing to the ML-TSSP HUMINT Dashboard! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

---

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, gender identity, sexual orientation, disability, personal appearance, race, ethnicity, age, religion, or nationality.

### Expected Behavior

- Be respectful and considerate in communication
- Welcome newcomers and help them get started
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling, insulting, or derogatory remarks
- Public or private harassment
- Publishing others' private information
- Other conduct that could reasonably be considered inappropriate

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Basic understanding of:
  - Streamlit
  - Machine Learning (TensorFlow, XGBoost)
  - Optimization (Pyomo)
  - Data visualization (Plotly, Matplotlib)

### First-Time Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/humint-dashboard.git
   cd humint-dashboard
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-owner/humint-dashboard.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## 💻 Development Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Set Up Pre-commit Hooks

```bash
pre-commit install
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your local settings
```

### 4. Run Tests

```bash
pytest tests/
```

### 5. Start Development Server

```bash
streamlit run dashboard.py
```

---

## 🔧 How to Contribute

### Types of Contributions

1. **Bug Reports**: Found a bug? Open an issue with details
2. **Feature Requests**: Have an idea? Discuss it in GitHub Discussions first
3. **Code Contributions**: Fix bugs or implement features
4. **Documentation**: Improve README, add examples, fix typos
5. **Testing**: Add or improve test coverage
6. **Design**: Improve UI/UX

### Finding Something to Work On

- Check [Issues](https://github.com/yourusername/humint-dashboard/issues) labeled `good first issue`
- Look for `help wanted` labels
- Review the [Roadmap](README.md#roadmap) section

### Before You Start

1. **Check existing issues** to avoid duplicates
2. **Discuss major changes** in an issue first
3. **Keep changes focused** - one feature/fix per PR
4. **Update documentation** if needed

---

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

```python
# Good
def calculate_source_score(reliability, trend, deception_risk):
    """
    Calculate composite source score based on multiple factors.
    
    Args:
        reliability (float): Source reliability score (0-1)
        trend (float): Historical trend indicator
        deception_risk (float): Deception probability (0-1)
    
    Returns:
        float: Composite score (0-12)
    """
    weights = {"reliability": 0.25, "trend": 0.15, "deception": 0.20}
    return (reliability * weights["reliability"] * 12 + 
            trend * weights["trend"] * 12 - 
            deception_risk * weights["deception"] * 12)
```

### Key Principles

1. **Clear naming**: Use descriptive variable and function names
2. **Docstrings**: All functions must have docstrings
3. **Type hints**: Use type hints for function parameters
4. **Comments**: Explain *why*, not *what*
5. **Error handling**: Use try-except blocks appropriately

### Code Formatting

We use `black` for code formatting:

```bash
# Format your code
black dashboard.py

# Check formatting
black --check dashboard.py
```

### Linting

We use `flake8` and `pylint`:

```bash
# Run linters
flake8 dashboard.py
pylint dashboard.py
```

### Import Organization

```python
# Standard library imports
import os
import sys
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import streamlit as st

# Local imports
from api import run_optimization
from utils import calculate_score
```

---

## 🧪 Testing Guidelines

### Writing Tests

All new features must include tests:

```python
# tests/test_optimization.py
import pytest
from optimization import calculate_emv

def test_calculate_emv_basic():
    """Test EMV calculation with basic inputs."""
    assignments = {"S1": "T1", "S2": "T2"}
    behaviors = {"S1": "Cooperative", "S2": "Cooperative"}
    task_values = {"T1": 1000, "T2": 1500}
    
    result = calculate_emv(assignments, behaviors, task_values)
    
    assert result > 0
    assert isinstance(result, float)

def test_calculate_emv_with_deception():
    """Test EMV calculation with deceptive sources."""
    # Test implementation
    pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_optimization.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run tests matching pattern
pytest -k "test_emv"
```

### Test Coverage

- Aim for **>80% code coverage**
- Critical paths require **100% coverage**
- Include edge cases and error conditions

---

## 📚 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def optimize_task_assignment(sources, tasks, constraints):
    """
    Optimize HUMINT source task assignments using ML-TSSP.
    
    This function implements two-stage stochastic programming to find
    optimal source-task assignments that maximize expected value while
    managing uncertainty in source behavior.
    
    Args:
        sources (List[Dict]): List of source profiles with features
        tasks (List[Dict]): List of tasks with requirements and values
        constraints (Dict): Optimization constraints including:
            - max_sources_per_task (int): Maximum sources per task
            - min_reliability (float): Minimum reliability threshold
            - budget (float): Available budget
    
    Returns:
        Dict: Optimization results containing:
            - assignments (Dict[str, str]): Source-to-task mapping
            - emv (float): Expected Monetary Value
            - vss (float): Value of Stochastic Solution
            - computation_time (float): Time taken in seconds
    
    Raises:
        ValueError: If inputs are invalid or optimization is infeasible
        RuntimeError: If solver fails to converge
    
    Example:
        >>> sources = [{"id": "S1", "reliability": 0.85, ...}]
        >>> tasks = [{"id": "T1", "value": 1000, ...}]
        >>> constraints = {"max_sources_per_task": 3, ...}
        >>> result = optimize_task_assignment(sources, tasks, constraints)
        >>> print(f"EMV: ${result['emv']:.2f}")
        EMV: $3850.00
    
    Note:
        This function may take several seconds for large problem instances.
        Consider using timeout parameter for production deployments.
    """
    # Implementation
    pass
```

### Updating README

When adding features, update:
- Features list
- Configuration section
- Usage examples
- Changelog

---

## 🔄 Pull Request Process

### 1. Prepare Your Changes

```bash
# Ensure your branch is up to date
git fetch upstream
git rebase upstream/main

# Run tests
pytest

# Run linters
black .
flake8 .

# Update documentation if needed
```

### 2. Commit Your Changes

Use conventional commit messages:

```bash
# Format: <type>(<scope>): <subject>

git commit -m "feat(optimization): add multi-objective optimization"
git commit -m "fix(dashboard): resolve login redirect issue"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(api): add tests for endpoint validation"
```

**Commit types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### 3. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 4. Create Pull Request

1. Go to GitHub and create a Pull Request
2. Fill out the PR template completely
3. Link related issues
4. Request review from maintainers

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex logic
- [ ] Updated documentation
- [ ] No breaking changes (or documented)
```

### 5. Review Process

- Maintainers will review within 2-3 business days
- Address feedback promptly
- Keep discussion professional and constructive
- Be patient - reviews take time!

### 6. After Merge

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Delete feature branch
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

---

## 🏆 Recognition

Contributors will be:
- Listed in the README
- Mentioned in release notes
- Credited in documentation

---

## ❓ Questions?

- Open an issue for technical questions
- Use GitHub Discussions for general questions
- Email maintainers for sensitive matters

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to ML-TSSP HUMINT Dashboard! 🚀**
