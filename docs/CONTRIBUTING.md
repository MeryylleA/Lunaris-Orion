# Contributing to Lunaris-Orion

First off, thank you for considering contributing to Lunaris-Orion! It's people like you that make Lunaris-Orion such a great tool for the pixel art and deep learning community.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How Can I Contribute?](#how-can-i-contribute)
4. [Development Process](#development-process)
5. [Pull Request Process](#pull-request-process)
6. [Style Guidelines](#style-guidelines)
7. [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [project maintainers].

## Getting Started

1. **Fork the Repository**
   - Click the Fork button in the top right corner of the repository page
   - Clone your fork locally: `git clone https://github.com/MeryylleA/Lunaris-Orion.git`

2. **Set Up Development Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows

   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number
   ```

## How Can I Contribute?

### Reporting Bugs
1. Check the issue tracker to avoid duplicates
2. Use the bug report template
3. Include:
   - Python version
   - PyTorch version
   - Operating system
   - Clear steps to reproduce
   - Expected vs actual behavior
   - Relevant logs or screenshots

### Suggesting Enhancements
1. Check existing enhancement requests
2. Use the feature request template
3. Provide:
   - Clear use case
   - Expected behavior
   - Potential implementation approach

### Code Contributions
1. **Architecture Improvements**
   - Attention mechanism optimizations
   - New layer implementations
   - Performance enhancements

2. **Training Improvements**
   - Loss function variations
   - Training stability enhancements
   - Memory optimization

3. **Documentation**
   - Code documentation
   - Usage examples
   - Architecture explanations

## Development Process

1. **Choose an Issue**
   - Look for issues labeled "good first issue" or "help wanted"
   - Comment on the issue to avoid duplicate work
   - Get assigned by a maintainer

2. **Development Guidelines**
   - Follow the architecture documentation
   - Maintain backward compatibility
   - Add tests for new features
   - Update documentation

3. **Testing**
   ```bash
   # Run tests
   pytest tests/
   
   # Run linting
   flake8 .
   black .
   isort .
   ```

## Pull Request Process

1. **Before Submitting**
   - Update documentation
   - Add tests
   - Run the test suite
   - Update requirements if needed
   - Follow code style guidelines

2. **PR Template**
   - Clear description of changes
   - Reference related issues
   - List breaking changes
   - Include test results

3. **Review Process**
   - Two approvals required
   - All tests must pass
   - Documentation must be updated
   - Code style must be consistent

## Style Guidelines

### Python Code Style
- Follow PEP 8
- Use Black for formatting
- Maximum line length: 88 characters
- Use type hints
- Document all public functions

### Documentation Style
- Clear and concise
- Include code examples
- Follow Google docstring format
- Update architecture.md for significant changes

### Commit Messages
```
type(scope): Brief description

Detailed description of changes
- Change 1
- Change 2

Fixes #issue_number
```

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Adding tests
- chore: Maintenance

## Community

### Communication Channels
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and discussions
- Pull Requests: Code review discussions

### Recognition
- Contributors are listed in CONTRIBUTORS.md
- Significant contributions are highlighted in release notes
- Community spotlight in documentation

## Additional Resources

1. **Documentation**
   - [Architecture Documentation](architecture.md)

2. **Tools**
   - PyTorch Documentation
   - Black Documentation
   - pytest Documentation

Thank you for contributing to Lunaris-Orion! Your efforts help make pixel art generation and manipulation more accessible to everyone. 
