# Contributing to PolyglotLink

Thank you for your interest in contributing to PolyglotLink! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/polyglotlink.git
   cd polyglotlink
   ```
3. Add the upstream repository as a remote:
   ```bash
   git remote add upstream https://github.com/polyglotlink/polyglotlink.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (for running infrastructure services)
- Git

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   make install-dev
   ```
   Or manually:
   ```bash
   pip install -e ".[dev,test]"
   pre-commit install
   ```

3. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

4. Start infrastructure services (Redis, MQTT broker, etc.):
   ```bash
   make docker-up
   ```

## Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. Make your changes, following our [code style guidelines](#code-style)

3. Write or update tests as needed

4. Run the test suite to ensure everything passes:
   ```bash
   make test
   ```

5. Commit your changes using [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add new protocol handler for Zigbee"
   git commit -m "fix: resolve MQTT connection timeout issue"
   git commit -m "docs: update installation instructions"
   ```

## Code Style

We use several tools to maintain code quality:

### Linting and Formatting

- **Ruff** for linting and formatting
- **MyPy** for static type checking

Run all checks:
```bash
make static-analysis
```

Or individually:
```bash
make lint        # Run linter
make lint-fix    # Auto-fix linting issues
make format      # Format code
make type-check  # Run type checking
```

### Style Guidelines

- Use type hints for all function parameters and return values
- Maximum line length is 100 characters
- Use descriptive variable and function names
- Write docstrings for public functions and classes
- Follow PEP 8 conventions

### Pre-commit Hooks

Pre-commit hooks run automatically on each commit. To run them manually:
```bash
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
make test              # Run all tests
make test-unit         # Run unit tests only
make test-integration  # Run integration tests
make test-cov          # Run tests with coverage report
make test-fast         # Run tests in parallel
```

### Writing Tests

- Place tests in `polyglotlink/tests/`
- Name test files with `test_` prefix
- Use pytest fixtures from `conftest.py`
- Aim for meaningful test coverage, not just high percentages
- Include both positive and negative test cases

### Test Categories

- **Unit tests**: Test individual functions and classes in isolation
- **Integration tests**: Test component interactions
- **System tests**: End-to-end workflow tests
- **Security tests**: Security-focused test cases
- **Performance tests**: Load and stress testing with Locust

## Submitting Changes

### Pull Request Process

1. Update documentation if needed
2. Ensure all tests pass
3. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
4. Open a Pull Request against the `main` branch
5. Fill out the PR template completely
6. Wait for CI checks to pass
7. Address any review feedback

### PR Guidelines

- Keep PRs focused on a single concern
- Write a clear description of what and why
- Link related issues using keywords (e.g., "Fixes #123")
- Include screenshots for UI changes
- Update the CHANGELOG.md for notable changes

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages
- Screenshots if applicable

### Feature Requests

When requesting features, please include:

- A clear description of the problem you're trying to solve
- Your proposed solution
- Alternative solutions you've considered
- Any additional context

## Questions?

If you have questions about contributing, feel free to:

- Open a discussion on GitHub
- Check existing issues and documentation
- Review the [FAQ](docs/FAQ.md)

Thank you for contributing to PolyglotLink!
