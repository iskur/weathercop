# GitHub Publication Readiness Audit Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Identify and resolve all issues preventing safe GitHub publication of the WeatherCop project.

**Architecture:** Systematic audit of repository contents, configuration, documentation, and code quality to identify publication blockers across multiple categories: security, legal, build quality, documentation, and repository hygiene.

**Tech Stack:** Python 3.13, Git, uv package manager, pytest, flake8, pre-commit

---

## ✅ RESOLVED: VARWG Library Now on PyPI

**STATUS:** Previously a blocker, now resolved! ✅

**Issue (Resolved):** WeatherCop previously depended on VG library only available via GitHub.

**Resolution:**
- VG library has been renamed to VARWG
- VARWG is now published on PyPI
- `pyproject.toml` should be updated to use `varwg` instead of git URL
- This unblocks PyPI publication of WeatherCop

**Impact:**
- WeatherCop can be published to **GitHub** ✅
- WeatherCop can now be published to **PyPI** ✅ (blocker removed!)
- Users can install via standard `pip install weathercop` from PyPI
- Users can also use `uv` which supports both PyPI and git dependencies

**Action Items:**
1. Update `pyproject.toml` to use `varwg` from PyPI
2. Update documentation and README to reference VARWG
3. Update any hardcoded VG references in code to VARWG
4. Test PyPI installation path

---

## Critical Issues Identified

### 🔴 BLOCKER: Large Binary Files in Repository (849MB + 3.7GB)

**Issue:** Repository contains massive binary libraries and cache data that will cause GitHub publication to fail:
- `src/weathercop/libs/`: 849MB of Intel MKL shared libraries (`.so` files)
- `cache/`: 3.7GB of pickle files and weather station data
- `src/weathercop/ufuncs/`: 237MB of auto-generated Cython code
- `docs/`: 62MB including PDF documents

**Impact:**
- GitHub has 100MB file size limit; repositories >1GB are problematic
- Git will be extremely slow with these files tracked
- Clone times will be prohibitive for users
- Violates GitHub's recommended practices

**Required Actions:**
1. Remove `src/weathercop/libs/` from repository completely
2. Add cache and data directories to `.gitignore`
3. Document how users should obtain/build required libraries
4. Clean git history if these were previously committed

---

### 🔴 BLOCKER: Hardcoded Personal Paths

**Issue:** Multiple hardcoded paths containing personal information:
- `cop_conf.py:14`: `weathercop_dir = home / "Projects" / "WeatherCop"`
- `cop_conf.py:14`: References `/home/dirk/Projects/WeatherCop`
- `multisite.py`: `xds = xr.open_dataset("/home/dirk/data/opendata_dwd/multisite_testdata.nc")`
- `postdoc_conf.py`: `root = Path("/home/dirk/Diss/thesis")`

**Impact:**
- Code won't work for any other user
- Exposes personal directory structure
- Professional credibility issue

**Required Actions:**
1. Make all paths relative or configurable
2. Use environment variables or config files for data paths
3. Provide example configuration with sensible defaults

---

### 🔴 BLOCKER: 340 Untracked Files in Repository

**Issue:** Git status shows 340 untracked files, indicating:
- Incomplete `.gitignore` configuration
- Mix of generated files, cache data, and potentially sensitive content
- Unclear what should vs. shouldn't be in repository

**Files include:**
- Cache files: `cache/vgs_cache/**/*.pkl`, `*.she.dat`, `*.she.dir`, `*.she.bak`
- Generated code: `src/weathercop/ufuncs/*.pyx` (auto-generated)
- Documentation: `docs/*.pdf`, `docs/*.jpg`, PDFs from other authors
- Build artifacts: possibly in various locations
- Configuration: `code/vg_conf.py`, `src/weathercop/kll_vg_conf.py`, `src/weathercop/postdoc_conf.py`

**Impact:**
- Repository is messy and unprofessional
- May accidentally commit sensitive or copyrighted material
- Users will be confused about project structure

**Required Actions:**
1. Audit all 340 files individually
2. Update `.gitignore` comprehensively
3. Decide which files should be tracked
4. Remove personal configuration files

---

### 🟡 HIGH PRIORITY: Empty README.md

**Issue:** README.md is completely empty (0 bytes), while README.org contains full documentation.

**Impact:**
- GitHub won't show any project information
- No installation instructions visible
- Poor first impression for potential users

**Required Actions:**
1. Pre-commit hook exists but README.md wasn't generated
2. Run: `pandoc -f org -t gfm README.org -o README.md`
3. Commit the generated README.md
4. Verify pre-commit hook works

---

### 🟡 HIGH PRIORITY: Personal Email in README.org

**Issue:** `README.org:126` contains personal email: `32363199+iskur@users.noreply.github.com`

**Impact:**
- Privacy concern for public repository
- May receive spam/unwanted contact

**Required Actions:**
1. Decide on contact method for public project
2. Use GitHub username or noreply email instead
3. Update LICENSE and pyproject.toml consistently

---

### 🟡 HIGH PRIORITY: Copyright Material in docs/

**Issue:** Directory contains PDFs that may be copyrighted:
- `dependence modeling with archimedean copulas - nelsen.pdf`
- `Introduction to vine copulas - beamer presentation.pdf`
- `copula introduction presentation Trondheim06.pdf`
- `DFG_raincop_vg_final.pdf`
- Various academic presentations and papers

**Impact:**
- Copyright violation potential
- Could lead to DMCA takedown
- Legal liability

**Required Actions:**
1. Remove all third-party PDFs
2. Replace with citations/links to original sources
3. Keep only original work
4. Add `*.pdf` to `.gitignore` if not already present

---

### 🟡 HIGH PRIORITY: Missing Dependency Installation Instructions

**Issue:** Project depends on custom VG library from GitHub:
- `pyproject.toml:5`: `vg = { git = "https://github.com/iskur/vg" }`
- No documentation about VG library requirements
- Users won't know if VG is public or how to access it

**Impact:**
- Installation will fail
- Users can't use the package
- Incomplete documentation

**Required Actions:**
1. Verify VG repository is public
2. Document VG dependency clearly in README
3. Add troubleshooting section for common install issues
4. Consider adding VG as git submodule or document alternative

---

### 🟡 MEDIUM PRIORITY: Build System Complexity

**Issue:**
- First import takes 5-10 minutes (Cython compilation)
- Users must run `python setup.py build_ext --inplace` manually
- No pre-built wheels available
- 237MB of auto-generated Cython code in `ufuncs/`

**Impact:**
- Poor user experience
- May think package is broken/hanging
- Barrier to adoption

**Required Actions:**
1. Add prominent warning in README
2. Document pre-build step clearly
3. Consider pre-generating common ufuncs
4. Add build troubleshooting section

---

### 🟡 MEDIUM PRIORITY: Flake8 Violations

**Issue:** Multiple code quality issues:
- Invalid escape sequences in strings (`\c` should be `\\c` or raw string)
- Unused imports (F401)
- Lines too long (E501) - exceeding 79 character limit
- Whitespace issues (E203)
- Unused variables (F841)

**Impact:**
- Code quality concerns
- May hide real bugs
- Unprofessional appearance

**Required Actions:**
1. Fix invalid escape sequences (syntax warnings)
2. Remove unused imports
3. Fix line length violations
4. Clean up unused variables

---

### 🟢 LOW PRIORITY: Personal/Development Config Files

**Issue:** Config files with personal settings in repo:
- `code/vg_conf.py` (untracked)
- `src/weathercop/kll_vg_conf.py`
- `src/weathercop/postdoc_conf.py`
- `src/weathercop/changing_correlations.py` (experimental?)

**Impact:**
- Confusing for users
- Mix of library code and personal scripts
- Unclear what's part of the package

**Required Actions:**
1. Move personal configs to examples/ or remove
2. Document configuration approach
3. Provide template config file
4. Clarify package vs. research scripts

---

### 🟢 LOW PRIORITY: Minor TODO Comments

**Issue:** Two TODO comments in code:
- `copulae.py:1523`: Name lookup needed
- `multisite.py:1433`: First realization issue

**Impact:** Minor - shows work in progress

**Required Actions:**
1. Resolve TODOs or convert to GitHub issues
2. Document known issues in CHANGELOG or issues

---

### 🟢 LOW PRIORITY: No Contributing Guidelines

**Issue:** No CONTRIBUTING.md, CODE_OF_CONDUCT.md, or development docs

**Impact:**
- Unclear how others can contribute
- May receive low-quality PRs

**Required Actions:**
1. Add CONTRIBUTING.md if accepting contributions
2. Document development workflow
3. Add issue templates (optional)

---

### 🟢 LOW PRIORITY: Test Configuration

**Issue:** `pyproject.toml:67` has `addopts = "--pdb"` which drops into debugger on test failure

**Impact:**
- CI/CD will hang
- Confusing for contributors

**Required Actions:**
1. Make `--pdb` optional or document it
2. Ensure CI doesn't use it

---

## Repository Structure Issues

### Missing Standard Files
- ✅ LICENSE exists (MIT)
- ❌ CONTRIBUTING.md missing
- ❌ CHANGELOG.md missing
- ❌ CITATION.cff missing (for academic software)
- ❌ .gitattributes missing (for proper LFS handling if needed)

---

## Implementation Tasks

### Task 1: Audit and Update .gitignore

**Files:**
- Modify: `.gitignore`

**Step 1: Add comprehensive ignores**

Append to `.gitignore`:

```gitignore
# Large binary libraries (should never be in repo)
src/weathercop/libs/

# Data files and caches (user-specific)
cache/
*.pkl
*.npy
ranks.npy

# Personal config files
code/vg_conf.py
src/weathercop/kll_vg_conf.py
src/weathercop/postdoc_conf.py

# Documentation - third-party PDFs
docs/*.pdf
docs/*.jpg
docs/*.png

# Auto-generated Cython code (should be regenerated on install)
# NOTE: Keeping these for now, but consider if they should be generated
# src/weathercop/ufuncs/*.pyx

# Build artifacts
*.egg-info/
.eggs/
```

**Step 2: Verify ignores**

Run: `git status --porcelain | wc -l`
Expected: Significantly fewer than 340 untracked files

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: update .gitignore for GitHub publication

- Exclude large binary libraries (849MB MKL files)
- Ignore cache and data files (3.7GB)
- Exclude personal configuration files
- Ignore third-party PDFs and images"
```

---

### Task 2: Remove Personal Paths from cop_conf.py

**Files:**
- Modify: `src/weathercop/cop_conf.py`

**Step 1: Make paths configurable**

Replace hardcoded paths with environment variable fallbacks:

```python
"""Central file to keep the same configuration in all weathercop code."""

from pathlib import Path
import multiprocessing
import os

# from dask.distributed import Client
from weathercop.tools import ADict

# setting PROFILE to True disables parallel computation, allowing for
# profiling and debugging
PROFILE = False
DEBUG = False

# Use environment variables or fallback to reasonable defaults
home = Path.home()
weathercop_dir = Path(os.environ.get(
    'WEATHERCOP_DIR',
    home / '.weathercop'
))
ensemble_root = Path(os.environ.get(
    'WEATHERCOP_ENSEMBLE_ROOT',
    weathercop_dir / "ensembles"
))

# Ensure directories exist
weathercop_dir.mkdir(exist_ok=True, parents=True)
ensemble_root.mkdir(exist_ok=True, parents=True)

script_home = Path(__file__).parent
ufunc_tmp_dir = script_home / "ufuncs"
sympy_cache = ufunc_tmp_dir / "sympy_cache.she"
cache_dir = weathercop_dir / "cache"
theano_cache = ufunc_tmp_dir / "theano_cache.she"
vine_cache = cache_dir / "vine_cache.she"
vgs_cache_dir = cache_dir / "vgs_cache"

varnames = ("R", "theta", "Qsw", "ILWR", "rh", "u", "v")
n_nodes = multiprocessing.cpu_count() - 2
n_digits = 5  # for ensemble realization file naming

# Create cache directories
cache_dir.mkdir(exist_ok=True, parents=True)
vgs_cache_dir.mkdir(exist_ok=True, parents=True)
```

**Step 2: Test configuration**

Run: `python -c "from weathercop import cop_conf; print(cop_conf.weathercop_dir)"`
Expected: Prints path to `~/.weathercop`

**Step 3: Commit**

```bash
git add src/weathercop/cop_conf.py
git commit -m "fix: make paths configurable via environment variables

- Replace hardcoded /home/dirk/Projects/WeatherCop paths
- Use WEATHERCOP_DIR environment variable with fallback to ~/.weathercop
- Ensure all directories are created on import
- Improves portability for all users"
```

---

### Task 3: Fix Hardcoded Path in multisite.py

**Files:**
- Modify: `src/weathercop/multisite.py`

**Step 1: Find and remove hardcoded test path**

Search for the line:
```python
xds = xr.open_dataset("/home/dirk/data/opendata_dwd/multisite_testdata.nc")
```

This should be either:
1. Removed if it's debug code
2. Converted to use a configurable path
3. Moved to test fixtures if it's test-related

If it's example/debug code, comment it out with explanation:

```python
# Example usage - replace with your own data path:
# xds = xr.open_dataset("path/to/your/multisite_testdata.nc")
```

**Step 2: Search for other hardcoded paths**

Run: `grep -n "/home/dirk" src/weathercop/multisite.py`
Expected: No matches

**Step 3: Test import**

Run: `python -c "from weathercop import multisite; print('OK')"`
Expected: No errors, prints OK

**Step 4: Commit**

```bash
git add src/weathercop/multisite.py
git commit -m "fix: remove hardcoded personal paths from multisite.py

- Remove /home/dirk/data path reference
- Add comment showing example usage pattern"
```

---

### Task 4: Handle Personal Config Files

**Files:**
- Remove: `src/weathercop/postdoc_conf.py` (or move to examples)
- Remove: `src/weathercop/kll_vg_conf.py` (or move to examples)
- Verify: These aren't imported by core library code

**Step 1: Check if files are imported**

Run: `grep -r "postdoc_conf\|kll_vg_conf" src/weathercop/*.py`
Expected: No imports in core library (only in test/example files is OK)

**Step 2: Create examples directory**

```bash
mkdir -p examples/configurations
```

**Step 3: Move personal configs**

If the configs are useful as examples:
```bash
git mv src/weathercop/postdoc_conf.py examples/configurations/
git mv src/weathercop/kll_vg_conf.py examples/configurations/
```

Or remove them entirely:
```bash
rm src/weathercop/postdoc_conf.py src/weathercop/kll_vg_conf.py
```

**Step 4: Create example config template**

Create: `examples/configurations/example_vg_conf.py`

```python
"""
Example VG configuration for WeatherCop.

Copy this file and customize for your own weather data setup.
"""
from pathlib import Path

# Example paths - customize these for your setup
root = Path("/path/to/your/data")
stations_file = root / "stations.csv"
weather_data_dir = root / "weather_data"

# Add your VG-specific configuration here
# See https://github.com/iskur/vg for VG documentation
```

**Step 5: Commit**

```bash
git add examples/
git commit -m "refactor: move personal configs to examples

- Move postdoc_conf.py and kll_vg_conf.py to examples/
- Add example_vg_conf.py template for users
- Personal configs not needed in core library"
```

---

### Task 5: Generate README.md from README.org

**Files:**
- Modify: `README.md` (generated from README.org)
- Modify: `README.org` (remove personal email)

**Step 1: Update README.org to remove personal email**

Replace in `README.org` line 126:
```org
Copyright (c) iskur <32363199+iskur@users.noreply.github.com>
```

With:
```org
Copyright (c) iskur <32363199+iskur@users.noreply.github.com>
```

**Step 2: Add installation troubleshooting section**

Add to README.org before "Running Tests":

```org
* Troubleshooting

** MKL Libraries Required

WeatherCop uses Intel MKL for performance-critical operations. If you encounter errors related to MKL libraries:

1. Install MKL via conda/mamba:
   #+begin_src bash
   conda install mkl
   # or
   mamba install mkl
   #+end_src

2. Or install via pip:
   #+begin_src bash
   pip install mkl
   #+end_src

** VG Library Dependency

WeatherCop requires the VG (Weather Generator) library. This is automatically installed from GitHub during =uv sync=, but you need:

- Git installed and accessible
- Internet connection during installation
- VG repository must be public or you need access credentials

If VG installation fails, check that you can access: https://github.com/iskur/vg

** First Import Takes 5-10 Minutes

The first time you import weathercop, Cython extensions compile automatically. This is normal. To avoid the wait:

#+begin_src bash
python setup.py build_ext --inplace
#+end_src

** Environment Variables

WeatherCop respects these environment variables:

- =WEATHERCOP_DIR=: Base directory for cache and data (default: =~/.weathercop=)
- =WEATHERCOP_ENSEMBLE_ROOT=: Directory for ensemble outputs (default: =$WEATHERCOP_DIR/ensembles=)
```

**Step 3: Generate README.md**

Run: `pandoc -f org -t gfm README.org -o README.md`
Expected: README.md created with full content

**Step 4: Verify README.md**

Run: `wc -l README.md`
Expected: Substantial file (>100 lines)

**Step 5: Commit**

```bash
git add README.org README.md
git commit -m "docs: update README with GitHub-ready content

- Remove personal email, use GitHub noreply address
- Add comprehensive troubleshooting section
- Document MKL and VG dependencies
- Add environment variable documentation
- Generate README.md from README.org"
```

---

### Task 6: Fix Code Quality Issues (flake8)

**Files:**
- Modify: `src/weathercop/copulae.py`
- Modify: `src/weathercop/changing_correlations.py`
- Modify: `src/weathercop/cop_conf.py`

**Step 1: Fix invalid escape sequences**

Find and fix lines with invalid escape sequences (the `SyntaxWarning` issues).
These are likely in docstrings or strings. Change `\c` to `\\c` or use raw strings `r"..."`.

Run flake8 to find exact lines:
```bash
python -W error::SyntaxWarning -c "import weathercop.copulae" 2>&1 | head -20
```

**Step 2: Remove unused imports**

In `src/weathercop/changing_correlations.py`:
```python
# Remove:
# from functools import partial
# from scipy import integrate
```

In `src/weathercop/cop_conf.py`:
```python
# Remove or comment out:
# from weathercop.tools import ADict
```

**Step 3: Fix line length violations**

For each E501 violation, break long lines:
```python
# Before (>79 chars):
some_very_long_function_call(argument1, argument2, argument3, argument4)

# After (<79 chars):
some_very_long_function_call(
    argument1, argument2, argument3, argument4
)
```

**Step 4: Remove unused variables**

In `src/weathercop/copulae.py` around line 1874-1875:
```python
# Remove or use:
# xx = ...
# yy = ...
```

**Step 5: Run flake8**

Run: `flake8 src/weathercop/*.py | wc -l`
Expected: Significantly fewer errors

**Step 6: Commit**

```bash
git add src/weathercop/*.py
git commit -m "style: fix flake8 violations for code quality

- Fix invalid escape sequences (SyntaxWarning)
- Remove unused imports
- Fix line length violations (E501)
- Remove unused variables
- Clean up whitespace issues (E203)"
```

---

### Task 7: Remove Third-Party PDFs from docs/

**Files:**
- Remove: `docs/*.pdf` (third-party papers)
- Create: `docs/REFERENCES.md`

**Step 1: Create references file**

Create: `docs/REFERENCES.md`

```markdown
# References and Resources

This document lists academic papers and resources relevant to WeatherCop.

## Copula Theory

- Nelsen, R.B. "Dependence Modeling with Archimedean Copulas"
  - Available at: [publisher link or DOI]

- Czado, C. "Introduction to Vine Copulas"
  - Available at: [publisher link or DOI]

## Vine Copula Methods

- [Add proper citations for vine copula methodology]

## Weather Generation

- [Add citations for weather generation methods]

## Related Software

- VG Weather Generator: https://github.com/iskur/vg

## Academic Presentations

For presentations and teaching materials, please refer to the original sources
or contact the authors directly.
```

**Step 2: Remove third-party PDFs**

Since these are untracked, just add to .gitignore (already done in Task 1).

If any PDFs are tracked in git:
```bash
git rm docs/*.pdf
```

**Step 3: Keep only original work**

Keep any diagrams/images you created yourself. Remove others.

**Step 4: Commit**

```bash
git add docs/REFERENCES.md
git commit -m "docs: replace third-party PDFs with reference list

- Remove copyrighted PDF documents
- Create REFERENCES.md with proper citations
- Avoid copyright issues for GitHub publication"
```

---

### Task 8: Document Library Dependencies

**Files:**
- Create: `docs/DEPENDENCIES.md`

**Step 1: Create dependency documentation**

Create: `docs/DEPENDENCIES.md`

```markdown
# WeatherCop Dependencies

## Core Dependencies

### Python Version
- **Python ≥ 3.13** required

### Package Manager
- **uv** (recommended): https://docs.astral.sh/uv/
- Alternative: pip + virtualenv

### External Libraries

#### VG (Weather Generator)
- **Source**: https://github.com/iskur/vg
- **Installation**: Automatic via `uv sync`
- **Purpose**: Time series analysis and weather generation
- **License**: [Check VG repository]
- **Note**: Must have git access to this repository

#### Intel MKL (Math Kernel Library)
- **Installation**:
  ```bash
  # Via conda/mamba:
  conda install mkl

  # Via pip:
  pip install mkl

  # Via system package manager (Arch Linux):
  yay -S intel-mkl
  ```
- **Purpose**: High-performance numerical computations
- **Size**: ~849MB of shared libraries
- **Note**: NOT included in repository, must be installed separately
- **Alternative**: OpenBLAS (not tested)

#### SLEEF (SIMD Library for Evaluating Elementary Functions)
- **Purpose**: Fast mathematical function evaluation
- **Installation**: Usually included with MKL or system libraries
- **Size**: ~20MB

### Build Dependencies

#### Cython
- **Version**: ≥ 3.1.2
- **Purpose**: Compile performance-critical code
- **Installation**: `uv sync --group dev`

#### NumPy
- **Version**: ≥ 2.3.1
- **Purpose**: Numerical arrays and build system
- **Installation**: Automatic via `uv sync`

### Runtime Dependencies

See `pyproject.toml` [project.dependencies] for complete list:
- cartopy (≥0.24.1) - Geospatial plotting
- xarray (≥2025.6.1) - Multi-dimensional data
- pandas (≥2.3.0) - Data manipulation
- scipy (≥1.16.0) - Scientific computing
- matplotlib (≥3.10.3) - Plotting
- sympy (≥1.14.0) - Symbolic mathematics (code generation)
- networkx (≥3.5) - Graph algorithms (vine structure)

## Development Dependencies

Install with: `uv sync --group dev`

- pytest (≥8.4.1) - Testing
- flake8 (≥7.3.0) - Linting
- black - Code formatting
- pre-commit (≥4.0.0) - Git hooks
- ipython (≥9.5.0) - Interactive development

## Optional Dependencies

Install with: `uv sync --group optional`

- pyfftw (≥0.15.0) - Fast Fourier transforms

## Platform Notes

### Linux
- Tested on Arch Linux (kernel 6.17.2)
- Should work on any modern Linux distribution

### macOS
- Not tested - contributions welcome

### Windows
- Not tested - contributions welcome
- MKL installation may differ

## Troubleshooting

### "Cannot find MKL libraries"

Ensure MKL is installed and libraries are in your system's library path:

```bash
# Check if MKL libraries exist:
ldconfig -p | grep mkl

# On Linux, you may need to set LD_LIBRARY_PATH:
export LD_LIBRARY_PATH=/path/to/mkl/lib:$LD_LIBRARY_PATH
```

### "Cannot install VG from GitHub"

1. Verify you can access: https://github.com/iskur/vg
2. Check git is installed: `git --version`
3. If VG is private, configure git credentials
4. Try manual installation:
   ```bash
   git clone https://github.com/iskur/vg
   cd vg
   pip install -e .
   ```

### "Cython compilation failed"

Ensure you have:
- C compiler (gcc/clang on Linux, MSVC on Windows)
- Python development headers
- NumPy installed

```bash
# Debian/Ubuntu:
sudo apt install python3-dev build-essential

# Arch Linux:
sudo pacman -S base-devel
```
```

**Step 2: Commit**

```bash
git add docs/DEPENDENCIES.md
git commit -m "docs: add comprehensive dependency documentation

- Document MKL installation requirements
- Explain VG library dependency
- Add troubleshooting for common install issues
- Note that 849MB of libraries NOT in repo
- Platform-specific installation notes"
```

---

### Task 9: Add CHANGELOG.md

**Files:**
- Create: `CHANGELOG.md`

**Step 1: Create changelog**

Create: `CHANGELOG.md`

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub publication readiness improvements
- Comprehensive dependency documentation
- Environment variable configuration support
- Troubleshooting documentation

### Changed
- Paths now configurable via environment variables instead of hardcoded
- Personal configuration files moved to examples/
- README generated from README.org with publication-ready content

### Fixed
- Code style violations (flake8 compliance)
- Invalid escape sequences in strings
- Removed personal paths from source code

### Removed
- Third-party PDF documents (replaced with citation list)
- Large binary libraries from repository (must install separately)
- Personal configuration files from source tree

## [0.1.0] - 2025-10-16

### Added
- Initial public release
- C-vine and R-vine copula implementations
- Seasonal copula with Fourier parameter smoothing
- Multisite weather generation workflows
- Automatic Cython code generation via SymPy
- Integration with VG library for temporal structure
- Comprehensive copula family library (Clayton, Gumbel, Joe, Plackett, etc.)
- Parallel processing support for ensemble generation
- Visualization tools for ensemble statistics

### Documentation
- README with installation and quick start guide
- Example configurations
- API documentation in docstrings

### Requirements
- Python ≥ 3.13
- Intel MKL or compatible BLAS library
- VG library (https://github.com/iskur/vg)

## Notes

### Migration to Public GitHub

This project was previously private/local development. Version 0.1.0 represents
the first public release with cleaned repository, proper documentation, and
resolved path/dependency issues.

### Known Issues

- First import takes 5-10 minutes while Cython extensions compile
- Large ufuncs directory (237MB of auto-generated code)
- MKL libraries required but not included (must install separately)
- VG library dependency requires GitHub access

### Future Plans

- Pre-built wheels for common platforms
- Reduce compilation time on first import
- Extended platform testing (macOS, Windows)
- Additional copula families
- Performance optimizations
```

**Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG documenting release history

- Document 0.1.0 initial public release
- Track changes for GitHub publication readiness
- Follow Keep a Changelog format
- Note known issues and migration from private dev"
```

---

### Task 10: Verify VG Dependency Accessibility

**Files:**
- Verify: VG repository accessibility
- Update: README.org if needed

**Step 1: Test VG repository access**

Run: `git ls-remote https://github.com/iskur/vg HEAD`

Expected: Success with commit hash, OR 404/access denied

**Step 2: If VG is private**

Update README.org and pyproject.toml to note this.

Add to README.org:

```org
** Important Note About VG Dependency

WeatherCop depends on the VG (Weather Generator) library, which is currently
hosted in a private repository. To use WeatherCop, you need:

1. Access to https://github.com/iskur/vg (request from maintainer)
2. Git credentials configured for GitHub access

Alternatively, wait for VG to be published publicly, or contact the maintainer
about access.
```

**Step 3: If VG is public**

Verify it works:
```bash
uv sync
```
Expected: VG installs successfully

**Step 4: Document findings**

Add note to docs/DEPENDENCIES.md about VG access requirements.

**Step 5: Commit**

```bash
git add README.org docs/DEPENDENCIES.md
git commit -m "docs: clarify VG dependency access requirements

- Note VG repository accessibility status
- Add instructions for requesting access if private
- Update dependency documentation"
```

---

### Task 11: Clean Up Test Configuration

**Files:**
- Modify: `pyproject.toml`
- Create: `pytest.ini` (optional alternative)

**Step 1: Make --pdb optional**

In `pyproject.toml`, change:

```toml
[tool.pytest.ini_options]
# Drop into debugger on failure (useful for development)
# Remove or comment out for CI/CD environments
# addopts = "--pdb"
```

**Step 2: Document test usage**

Add to README.org in "Running Tests" section:

```org
* Running Tests

To run the test suite:

#+begin_src bash
uv run pytest
#+end_src

For development with automatic debugger on failures:

#+begin_src bash
uv run pytest --pdb
#+end_src

Or install test dependencies and run:

#+begin_src bash
uv sync --group test
uv run pytest
#+end_src

*Note*: The =--pdb= flag is commented out by default in =pyproject.toml=.
Uncomment it for interactive debugging during development, but keep it
disabled for CI/CD.
```

**Step 3: Regenerate README.md**

Run: `pandoc -f org -t gfm README.org -o README.md`

**Step 4: Commit**

```bash
git add pyproject.toml README.org README.md
git commit -m "fix: make pytest --pdb optional for CI compatibility

- Comment out --pdb in pyproject.toml by default
- Document how to enable for development
- Prevent CI/CD hangs on test failures"
```

---

### Task 12: Add CONTRIBUTING.md (Optional)

**Files:**
- Create: `CONTRIBUTING.md`

**Step 1: Create contributing guide**

Create: `CONTRIBUTING.md`

```markdown
# Contributing to WeatherCop

Thank you for your interest in contributing to WeatherCop!

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/weathercop
   cd weathercop
   ```

3. Install development dependencies:
   ```bash
   uv sync --group dev
   python setup.py build_ext --inplace
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow PEP 8 style guidelines (79 character line limit)
- Add docstrings for new functions/classes
- Update tests for changed functionality
- Update documentation if needed

### 3. Run Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest src/weathercop/tests/test_vine.py

# Run with coverage
uv run pytest --cov=weathercop
```

### 4. Check Code Quality

```bash
# Lint code
flake8 src/weathercop/

# Format code
black --line-length 79 src/

# Check formatting without changes
black --check --line-length 79 src/
```

### 5. Commit Changes

```bash
git add .
git commit -m "type: brief description

Longer explanation if needed.

Fixes #issue-number"
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting
- `refactor`: Code restructuring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## What to Contribute

### Good First Issues

- Documentation improvements
- Test coverage additions
- Bug fixes with clear reproduction steps
- Performance optimizations with benchmarks

### Larger Contributions

Please open an issue first to discuss:
- New copula families
- API changes
- New major features
- Significant refactoring

## Code Style

- **Line length**: 79 characters (configured in pyproject.toml)
- **Formatter**: Black with `--line-length 79`
- **Linter**: Flake8
- **Docstrings**: NumPy style
- **Type hints**: Encouraged but not required

## Testing

- Add tests for all new features
- Ensure existing tests pass
- Aim for >80% code coverage
- Use pytest fixtures for common setup

## Documentation

- Update README.org if changing user-facing features
- Run `pandoc -f org -t gfm README.org -o README.md` to generate README.md
- Update CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/)
- Add docstrings for public APIs

## Pre-commit Hooks

We use pre-commit for automated checks:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Questions?

- Open an issue for questions
- Check existing issues and discussions
- See docs/ directory for additional documentation

## License

By contributing, you agree that your contributions will be licensed under the
MIT License (see LICENSE file).
```

**Step 2: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add contributing guidelines

- Document development workflow
- Explain code style requirements
- Add testing and documentation guidelines
- Make project welcoming for contributors"
```

---

### Task 13: Final Repository Audit

**Files:**
- Verify all changes
- Check git status

**Step 1: Run final checks**

```bash
# Verify no sensitive paths remain
grep -r "/home/dirk" src/weathercop/*.py

# Check git status
git status

# Verify .gitignore works
git status --porcelain | wc -l

# Run tests
uv run pytest

# Check code quality
flake8 src/weathercop/*.py | wc -l
```

**Step 2: Verify README renders**

Check that README.md has content:
```bash
head -20 README.md
```

**Step 3: Check repository size**

```bash
# Check total size (should be <100MB without cache/libs)
du -sh .git
```

**Step 4: Create summary document**

Create: `docs/PUBLICATION_CHECKLIST.md`

```markdown
# GitHub Publication Checklist

This document tracks readiness for public GitHub publication.

## ✅ Completed

- [x] Removed 849MB of MKL libraries from repository
- [x] Removed 3.7GB cache directory from tracking
- [x] Updated .gitignore comprehensively
- [x] Removed personal hardcoded paths
- [x] Made configuration environment-variable based
- [x] Generated README.md from README.org
- [x] Removed personal email from public docs
- [x] Removed third-party copyrighted PDFs
- [x] Created reference list for citations
- [x] Fixed flake8 code quality violations
- [x] Added comprehensive dependency documentation
- [x] Created CHANGELOG.md
- [x] Added CONTRIBUTING.md
- [x] Documented VG dependency requirements
- [x] Made pytest --pdb optional
- [x] Moved personal configs to examples/

## ⚠️ Warnings / Notes

- First import takes 5-10 minutes (Cython compilation)
- 237MB of auto-generated ufuncs code in repository
- VG dependency requires GitHub access
- MKL libraries (849MB) must be installed separately
- Test suite uses --pdb which may confuse some users

## 📋 Pre-Publication Final Steps

Before pushing to public GitHub:

1. **Create clean branch**
   ```bash
   git checkout -b release/public-v0.1.0
   ```

2. **Verify no large files**
   ```bash
   find . -type f -size +10M | grep -v ".git"
   ```

3. **Test installation from scratch**
   ```bash
   cd /tmp
   git clone /path/to/weathercop test-install
   cd test-install
   uv sync --group dev
   python setup.py build_ext --inplace
   uv run pytest
   ```

4. **Create GitHub repository**
   - Go to github.com/new
   - Choose public visibility
   - Add description from README
   - Do NOT initialize with README (we have one)

5. **Push to GitHub**
   ```bash
   git remote add origin git@github.com:iskur/weathercop.git
   git push -u origin main
   ```

6. **Post-publication**
   - Add topics/tags for discoverability
   - Enable issues
   - Add repository description
   - Create v0.1.0 release
   - Consider adding DOI via Zenodo

## 🔍 Issues Not Addressed

These are acceptable for initial publication but should be tracked:

- [ ] Long compile time on first import (consider pre-built wheels)
- [ ] Large ufuncs directory (consider generation on install)
- [ ] Platform testing (only Linux tested)
- [ ] CI/CD setup (GitHub Actions)
- [ ] Code coverage reporting
- [ ] Comprehensive API documentation (consider Sphinx)
- [ ] Tutorial notebooks
- [ ] Example datasets (too large for repo, need external hosting)

## 📝 Ongoing Maintenance

After publication:

- Monitor issues for installation problems
- Document common issues in troubleshooting
- Consider adding GitHub Actions for CI
- Add badges to README (license, Python version, etc.)
- Set up GitHub Discussions for community
```

**Step 5: Commit**

```bash
git add docs/PUBLICATION_CHECKLIST.md
git commit -m "docs: add publication readiness checklist

- Track all completed publication prep tasks
- Document remaining warnings and notes
- Provide step-by-step publication guide
- List post-publication maintenance tasks"
```

---

## Summary

This plan addresses all identified blockers for GitHub publication:

### Critical Issues (MUST fix before publication)
1. ✅ Large binary files (849MB + 3.7GB) - excluded via .gitignore
2. ✅ Hardcoded personal paths - made configurable
3. ✅ 340 untracked files - comprehensive .gitignore update
4. ✅ Empty README.md - generated from README.org
5. ✅ Personal email exposure - replaced with GitHub noreply
6. ✅ Third-party copyrighted PDFs - removed, citations added

### High Priority (Should fix)
7. ✅ Missing dependency documentation - comprehensive docs added
8. ✅ Code quality issues - flake8 violations fixed
9. ✅ Personal config files - moved to examples/

### Medium Priority (Nice to have)
10. ✅ Test configuration (--pdb) - made optional
11. ✅ Contributing guidelines - added
12. ✅ Changelog - added
13. ✅ VG dependency clarity - documented

After completing these tasks, the repository will be ready for safe public publication on GitHub.

## Execution Handoff

Plan complete and saved to `docs/plans/2025-10-16-github-publication-readiness.md`.

Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach would you prefer?
