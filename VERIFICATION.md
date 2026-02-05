# Cross-Platform CI/CD Implementation Verification

This document provides step-by-step verification for the cross-platform CI/CD delegation plan implementation.

## Overview

The implementation automates cross-platform wheel building and testing by:
- **GitLab CI**: Runs comprehensive tests on Linux (primary validation)
- **GitHub Actions**: Builds wheels for all platforms (Ubuntu, Windows, macOS Intel, macOS ARM)
- **GitHub Actions**: Runs smoke tests on Windows/macOS (skips Linux - GitLab handles full testing)
- **Coordination**: GitLab triggers GitHub, waits for results, then publishes all wheels atomically

## Local Verification Steps

### Step 1: Verify pyproject.toml Configuration

```bash
# Check package-data configuration (should include *.nc files)
grep -A 5 "tool.setuptools.package-data" pyproject.toml
```

Expected output:
```toml
[tool.setuptools.package-data]
weathercop = [
    "tests/fixtures/*.nc",
    "tests/fixtures/.gitignore",
    "examples/*.py",
]
```

### Step 2: Verify Test Data File Exists

```bash
# Check that the multisite test data is in the project
ls -lh src/weathercop/tests/fixtures/multisite_testdata.nc

# Expected: ~4.1 MB file
```

### Step 3: Build Wheel Locally

```bash
# Install build tools
uv sync --group dev

# Pre-generate Cython ufuncs to avoid runtime compilation
python scripts/generate_ufuncs.py

# Build wheels
uv build --wheel

# List built wheels
ls -lh dist/*.whl
```

### Step 4: Verify .nc File is Packaged

```bash
# Verify the .nc test data file is in the wheel
python -m zipfile -l dist/*.whl | grep multisite_testdata.nc

# Should show:
# weathercop/tests/fixtures/multisite_testdata.nc

# For detailed verification:
python << 'EOF'
import zipfile
from pathlib import Path

for wheel in Path('dist').glob('*.whl'):
    with zipfile.ZipFile(wheel, 'r') as zf:
        nc_files = [f for f in zf.namelist() if f.endswith('.nc')]
        if nc_files:
            print(f"✓ {wheel.name}")
            for nc in nc_files:
                info = zf.getinfo(nc)
                print(f"  - {nc} ({info.file_size / (1024*1024):.2f} MB)")
        else:
            print(f"✗ {wheel.name}: No .nc files found!")
            exit(1)
EOF
```

### Step 5: Test Installation in Clean Virtualenv

```bash
# Create clean virtual environment
python -m venv /tmp/test_weathercop_env
source /tmp/test_weathercop_env/bin/activate  # On Windows: /tmp/test_weathercop_env\Scripts\activate

# Install wheel with minimal test dependencies
pip install dist/*.whl xarray netcdf4 matplotlib pandas scipy

# Verify example data path works
python -c "from weathercop.example_data import get_example_dataset_path; print(f'Data path: {get_example_dataset_path()}')"

# Expected: Should print path and NOT raise FileNotFoundError
```

### Step 6: Run Smoke Test Script

```bash
# Ensure you're in the clean virtualenv from Step 5

# Run all 4 test levels (comprehensive, ~20 minutes)
python scripts/smoke_test.py

# Run specific levels (faster):
python scripts/smoke_test.py --level 1  # Just imports (~2 min)
python scripts/smoke_test.py --level 2  # Imports + data file (~3 min)
python scripts/smoke_test.py --level 3  # Full + MultiSite (~15 min) - RECOMMENDED FOR LOCAL TESTING
```

Expected output:
```
======================================================================
  WeatherCop Smoke Tests (Levels 1-3)
======================================================================

======================================================================
  Level 1: Import Validation
======================================================================

Importing weathercop.copulae... OK
Importing weathercop.vine... OK
...
✓ Level 1 passed in X.Xs

======================================================================
  Level 2: Data File Validation
======================================================================

Getting example dataset path... OK (src/weathercop/tests/fixtures/multisite_testdata.nc)
...
✓ Level 2 passed in X.Xs

======================================================================
  Level 3: Core Functionality
======================================================================

Configuring VARWG... OK
...
✓ Level 3 passed in X.Xs

======================================================================
  Summary
======================================================================

Passed: 3/3 levels
Total time: X.Xs
✓ All smoke tests passed!
```

## CI/CD Verification Steps

### GitHub Actions Verification

1. **Watch build progress**: Monitor `.github/workflows/build-wheels.yml`
   - Builds should complete on all 4 platforms: Ubuntu, Windows, macOS Intel, macOS ARM
   - Verify smoke tests run on Windows/macOS (skip on Ubuntu with message about GitLab handling full tests)

2. **Check wheel artifacts**:
   - Windows wheel should be named: `weathercop-0.2.0-cp313-cp313-win_amd64.whl`
   - macOS Intel wheel: `weathercop-0.2.0-cp313-cp313-macosx_10_13_x86_64.whl`
   - macOS ARM wheel: `weathercop-0.2.0-cp313-cp313-macosx_11_0_arm64.whl`
   - Ubuntu wheel: `weathercop-0.2.0-cp313-cp313-manylinux_2_17_x86_64.whl`

3. **Verify .nc file in wheels**:
   - Each wheel should include `weathercop/tests/fixtures/multisite_testdata.nc`
   - GitHub Actions checks this with "Verify test data is packaged in wheel" step

4. **Monitor smoke test output**:
   - Look for Level 3 tests passing (Levels 1-3 run on Windows/macOS)
   - Should show "✓ All smoke tests passed!" for each platform

### GitLab CI Verification

1. **Full test suite**: Monitor `.gitlab-ci.yml` test-build job
   - Should run full pytest suite with all tests on Linux
   - Much slower than smoke tests but comprehensive

2. **Build wheels**: Monitor build-wheels job
   - Should build Linux wheel (manylinux)
   - Should verify .nc file is packaged

3. **Trigger GitHub**: Monitor trigger-github-builds job
   - Should trigger GitHub Actions workflow
   - Should poll and wait for GitHub to complete
   - Should fail if GitHub smoke tests fail (blocks publishing)

4. **Publishing**: Monitor publish job (manual approval)
   - Should combine Linux wheels from GitLab
   - Should download Windows/macOS wheels from GitHub artifacts
   - Should publish all wheels together to PyPI

## Expected Behavior

### On Tag Push (e.g., git tag v0.2.0)

**GitLab CI Pipeline:**
1. ✓ test-build: Runs full test suite (green ✓)
2. ✓ build-wheels: Builds Linux wheel, verifies .nc file (green ✓)
3. ✓ build-sdist: Builds source distribution (green ✓)
4. ⏳ trigger-github-builds: Waits for GitHub (status: waiting/polling)
5. ⏸ publish: Waits for approval (status: blocked, needs manual trigger)

**GitHub Actions Pipeline** (triggered by GitLab):
1. ✓ build_wheels: Builds for all 4 platforms
   - Windows: builds + smoke tests (Levels 1-3) ✓
   - macOS Intel: builds + smoke tests (Levels 1-3) ✓
   - macOS ARM: builds + smoke tests (Levels 1-3) ✓
   - Ubuntu: builds only, no tests (GitLab handles full testing)

2. ✓ build_sdist: Already built by GitLab, GitHub skips or redoes

3. ✓ Upload artifacts: Wheels uploaded to GitHub Actions artifacts

**After GitHub Completes:**
- GitLab trigger-github-builds job exits with success
- GitLab publish job becomes available for manual approval
- On approval, GitLab:
  - Downloads Windows/macOS wheels from GitHub artifacts
  - Combines with Linux wheels
  - Publishes all to PyPI (with --skip-existing for coordination)

### If Any Smoke Test Fails

1. GitHub job fails with message showing which platform/level failed
2. GitLab poll detects failure and exits with error
3. GitLab pipeline marked as failed
4. Publish job NOT available (no manual approval option)
5. PyPI upload does not proceed

## Troubleshooting

### Issue: ".nc file not found in wheel"

**Root cause**: setuptools not including package data properly

**Solutions** (in order of preference):
1. Verify `pyproject.toml` has correct package-data config
2. Add explicit include to `MANIFEST.in`:
   ```
   recursive-include src/weathercop/tests/fixtures *.nc
   ```
3. Move test data to different location (if other solutions fail):
   ```bash
   mkdir -p src/weathercop/data
   mv src/weathercop/tests/fixtures/multisite_testdata.nc src/weathercop/data/
   ```

### Issue: "GitHub workflow not triggered"

**Root cause**: Missing or invalid GITHUB_TOKEN

**Solution**:
1. Create GitHub Personal Access Token (PAT):
   - GitHub Settings → Developer Settings → Personal Access Tokens → Fine-grained tokens
   - Permissions needed: `actions:write`, `contents:read`
   - Copy the token
2. Add to GitLab CI variables:
   - GitLab Settings → CI/CD → Variables
   - Add: `GITHUB_TOKEN` = `<your-pat>`
   - Make it masked and protected
3. Update GITHUB_REPO in .gitlab-ci.yml if needed (currently "iskur/weathercop")

### Issue: "Smoke test timeout on GitHub"

**Solutions** (if total runtime exceeds 60 minutes):
1. Reduce test data further in smoke_test.py:
   ```python
   xds_small = xds.isel(station=slice(0, 1), time=slice(0, 365))  # 1 station, 1 year
   ```
2. Skip Level 4 (quick start) in GitHub, keep Levels 1-3:
   ```yaml
   CIBW_TEST_COMMAND: "python {project}/scripts/smoke_test.py --level 3"
   ```
3. Skip Apple Silicon if necessary:
   ```yaml
   CIBW_TEST_SKIP: "cp*-manylinux* cp*-musllinux* *-macosx_arm64"
   ```

### Issue: "Publish job appears stuck"

**Likely cause**: Waiting for manual approval

**Solution**: GitLab publish job has `when: manual` - you must click "Play" button in GitLab UI to trigger it

## Implementation Checklist

- [x] Phase 1: Add .nc verification to both CI configs
- [x] Phase 2: Create smoke_test.py with 4 levels
- [x] Phase 3: Configure GitHub Actions for smoke testing
- [x] Phase 4: Update GitLab CI publish command with --skip-existing
- [x] Phase 5: Add trigger-github-builds stage and job to GitLab CI
- [ ] Phase 6a: Local verification (run Steps 1-6 above)
- [ ] Phase 6b: Create test tag and verify CI behavior
- [ ] Phase 6c: Test production release workflow

## Creating Test Tag

```bash
# Create a test tag to verify CI without releasing
git tag v0.2.0-test1
git push origin v0.2.0-test1

# Monitor GitLab pipeline:
# https://gitlab.com/iskur/weathercop/-/pipelines

# After successful verification, delete test tag:
git tag -d v0.2.0-test1
git push origin :refs/tags/v0.2.0-test1
```

## Production Release Workflow

```bash
# 1. Create release tag
git tag v0.2.0
git push origin v0.2.0

# 2. Monitor GitLab pipeline:
# - Wait for trigger-github-builds to complete
# - Verify GitHub Actions created wheels for all platforms

# 3. Manually approve publish:
# - Go to GitLab pipeline
# - Click "Play" on the publish job
# - Verify wheels are uploaded to PyPI

# 4. Verify on PyPI:
pip install weathercop==0.2.0
python -c "from weathercop.example_data import get_example_dataset_path; print(get_example_dataset_path())"
```

## Files Modified

- **scripts/smoke_test.py** (NEW): 4-level smoke test script
- **pyproject.toml**: Added comment about test configuration
- **.github/workflows/build-wheels.yml**: Added smoke test configuration
- **.gitlab-ci.yml**:
  - Added `trigger-github` stage
  - Added `trigger-github-builds` job
  - Updated `build-wheels` with .nc verification
  - Updated `publish` with `--skip-existing`
  - Updated `publish` job with `needs` clause

## Timeline

- **Local verification**: ~30-45 minutes (one-time setup)
- **Test tag CI run**: ~90-120 minutes (parallel builds across platforms)
- **Production release**: ~120-150 minutes (full pipeline with all stages)
