# Contributing to pypi-mono

Thank you for your interest in contributing! 🎉

## Development Setup

1. **Fork and clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/pypi-mono.git
   cd pypi-mono
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or: venv\Scripts\activate  # Windows
   ```

3. **Install dev dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

## Code Standards

- **Linting**: We use `ruff` for linting
  ```bash
  ruff check .
  ```

- **Formatting**: Code should be formatted properly
  ```bash
  ruff format .
  ```

- **Testing**: Write tests for new features
  ```bash
  pytest
  ```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Ensure CI passes (tests + linting)
4. Request review from maintainers
5. Squash and merge after approval

## Branch Naming

- `feat/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `chore/` - Maintenance tasks

## Commit Messages

Follow conventional commits:
- `feat: add new feature`
- `fix: resolve issue with X`
- `docs: update README`
- `chore: update dependencies`

## Questions?

Open an issue or start a discussion!
