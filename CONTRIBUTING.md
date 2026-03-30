# Contributing to WeatherCop

Thank you for your interest in contributing to WeatherCop! We welcome contributions of all kinds—bug reports, feature requests, documentation improvements, and code contributions.

## How to Contribute

### Reporting Issues

Open an issue on [GitHub](https://github.com/iskur/weathercop/issues) to report bugs or suggest features.

### Submitting Pull Requests

We welcome pull requests on GitHub:

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and write clear commit messages
4. **Test locally** (see Development section below)
5. **Push to your fork** and open a Pull Request on GitHub
6. **Describe your changes** clearly in the PR description

## Development

### Setup

```bash
# Install development dependencies
uv sync --group dev

# Build Cython extensions in-place
python setup.py build_ext --inplace

# Pre-generate Cython ufuncs
python scripts/generate_ufuncs.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest src/weathercop/tests/test_vine.py

# Run with verbose output
pytest -v
```

### Code Style

We use:
- **Black** for formatting (line length: 79): `black --line-length 79 src/`
- **Flake8** for linting: `flake8 src/`
- **Type checking** is encouraged but not required

### Documentation

When adding features, please:
- Add docstrings to functions and classes
- Update relevant documentation
- Include examples for complex features

## Workflow

Our development workflow uses:
- **GitLab** (private): Primary development and testing
- **GitHub** (public): Public repository and releases
- **GitHub Actions**: Multi-platform wheel building (Windows, macOS)
- **GitLab CI**: Full testing and Linux wheel building

When your PR is approved:
1. We'll test it in our CI/CD pipeline
2. Merge it to our main branch
3. Sync the code to GitHub for releases

## Questions?

- Check the [README](README.md) for project overview
- Review the [CLAUDE.md](CLAUDE.md) for architecture details
- Open an issue for questions or discussions

Thank you for contributing! 🙏
