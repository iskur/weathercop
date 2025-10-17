# Changelog

All notable changes to WeatherCop will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- DEPENDENCIES.md documenting all library dependencies and their purpose
- REFERENCES.md with academic references for copula theory and vine copulas
- CHANGELOG.md for tracking project changes
- CLAUDE.md for AI assistant guidance on project structure
- Pre-commit hooks for automatic README.md generation from README.org
- LICENSE file (MIT)
- Troubleshooting section in README for common issues
- Environment variable configuration (WEATHERCOP_DIR, WEATHERCOP_ENSEMBLE_ROOT)
- Example configuration files in examples/configurations/

### Changed
- Made all paths configurable via environment variables instead of hardcoded paths
- Updated Python requirement to ≥3.13
- Updated all dependencies to latest versions (2025)
- Moved experimental code to docs/ directory
- Moved personal configuration files to examples/configurations/
- Improved .gitignore to exclude large binaries, cache files, and personal configs
- Updated contact email to GitHub noreply address for privacy

### Removed
- Third-party PDF files from docs/ (13MB) - now referenced in REFERENCES.md
- Unused MKL dependency and implementation variants
- Hardcoded personal paths from source code
- Personal configuration files from source tree

### Fixed
- Code style issues (flake8 violations)
- Hardcoded paths that prevented portability
- Unused imports in multiple files

## [0.1.0] - 2025-10-17

Initial development release prepared for GitHub publication.

### Core Features
- Canonical vine (CVine) and regular vine (RVine) copula implementations
- 15+ bivariate copula families (Archimedean, Elliptical, and others)
- Automatic copula selection via maximum likelihood
- Seasonal copula support with time-varying parameters
- Multisite weather generation with spatial dependence
- Phase randomization methods for temporal dependence
- Automatic Cython code generation from symbolic expressions
- Integration with VG library for marginals and missing values
- Geospatial visualization support via Cartopy
- Comprehensive test suite

### Known Issues
- VG dependency currently installed from GitHub (PyPI publication pending)
- First import takes 5-10 minutes for Cython compilation
- Some test failures marked as known issues (temp gradient with non-Gaussian marginals)

---

## Version History Notes

This project has been in active development since 2017. This changelog was created
during preparation for the first public release on GitHub. Earlier development history
is available in the git commit log.
