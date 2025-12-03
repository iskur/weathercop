# Changelog

All notable changes to WeatherCop will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-12-03

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
- Test fixture data for portable CI testing across environments
- pytest marker registration for `@pytest.mark.slow` tests
- Pytest parallelization with memory-aware phased execution
- Memory optimization tests with surgical attribute exclusion for multiprocessing
- Logging support for debugging in multisite weather generation
- Support for `xr.DataArray` in ensemble structure validation (in addition to `xr.Dataset`)

### Changed
- Made all paths configurable via environment variables instead of hardcoded paths
- Updated Python requirement to ≥3.13
- Updated all dependencies to latest versions (2025)
- Moved experimental code to docs/ directory
- Moved personal configuration files to examples/configurations/
- Improved .gitignore to exclude large binaries, cache files, and personal configs
- Updated contact email to GitHub noreply address for privacy
- Refactored test suite to use pytest fixtures and parametrization (test_vine.py, test_copulae.py)
- Improved Cython compilation pipeline with parallel compilation (nthreads=4)
- Enhanced test_density numerical integration to prevent CI timeouts
- Updated varwg dependency to version 1.4.2
- Reorganized configuration files for consistency
- Eliminated NumPy deprecation warnings in Newton's method
- Switched back to multiprocessing with surgical attribute exclusion for workers
- Increased CI test timeouts to handle resource constraints
- Made pytest execution sequential to prevent memory exhaustion in CI

### Fixed
- Code style issues (flake8 violations)
- Hardcoded paths that prevented portability
- Unused imports in multiple files
- CI race conditions in Cython ufunc generation by pre-compiling core extensions
- Memory exhaustion during parallel test execution in CI environments
- All-NaN ensemble generation issues through improved data validation
- Missing test data handling with graceful fallbacks
- Multiprocessing pickle serialization issues via surgical attribute exclusion
- CI build failures related to --system flag in uv pip install
- Assertion type testing accuracy

### Removed
- Third-party PDF files from docs/ (13MB) - now referenced in REFERENCES.md
- Unused MKL dependency and implementation variants
- Hardcoded personal paths from source code
- Personal configuration files from source tree
- `@pytest.mark.slow` annotation from ensemble fixtures (now properly registered)
- Test for non-primary variable simulation (behavior no longer supported)

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
