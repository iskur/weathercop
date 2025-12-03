# Pytest Parallelization Strategy for weathercop

## Project Context

**Repository**: `github.com/iskur/weathercop`  
**Testing Framework**: pytest  
**Goal**: Optimize test execution time while managing memory constraints

## Problem Statement

The weathercop test suite has:
- **CPU-intensive tests**: `test_copulae.py` with heavily parameterized tests that are computationally expensive but don't consume excessive memory
- **Memory-intensive tests**: `test_ensemble.py` and `test_multisite.py` that can cause OOM issues if run simultaneously
- Need to maximize parallelization for CPU-bound tests while preventing memory-intensive tests from running concurrently

## Solution: Phased Test Execution with pytest-xdist

### Core Strategy

Use pytest markers to categorize tests, then run them in two phases:
1. **Phase 1**: Run non-memory-intensive tests in full parallel (utilize all CPU cores)
2. **Phase 2**: Run memory-intensive tests sequentially or with limited parallelization

### Implementation Steps

#### 1. Install pytest-xdist

```bash
uv add --dev pytest-xdist
```

This will add pytest-xdist to your `pyproject.toml` dev dependencies.

#### 2. Mark Memory-Intensive Tests

Add markers to memory-intensive test modules:

**In `test_ensemble.py`:**
```python
import pytest

# Mark entire module
pytestmark = pytest.mark.memory_intensive
```

**In `test_multisite.py`:**
```python
import pytest

# Mark entire module
pytestmark = pytest.mark.memory_intensive
```

Alternatively, mark individual tests if only some tests are memory-intensive:
```python
@pytest.mark.memory_intensive
def test_large_ensemble_operation():
    ...
```

#### 3. Register the Marker

Add to `pytest.ini` or `pyproject.toml`:

**pytest.ini:**
```ini
[pytest]
markers =
    memory_intensive: marks tests as memory-intensive (deselect with '-m "not memory_intensive"')
```

**pyproject.toml:**
```toml
[tool.pytest.ini_options]
markers = [
    "memory_intensive: marks tests as memory-intensive (deselect with '-m \"not memory_intensive\"')",
]
```

#### 4. Local Testing Commands

```bash
# Run CPU-intensive tests in parallel (uses all available cores)
uv run pytest -n auto -m "not memory_intensive"

# Run memory-intensive tests sequentially
uv run pytest -n 1 -m "memory_intensive"

# Or combine into single command
uv run pytest -n auto -m "not memory_intensive" && uv run pytest -n 1 -m "memory_intensive"
```

**Alternative distribution strategies** (if needed):
```bash
# Group tests by file - keeps all tests from same module on same worker
uv run pytest -n auto --dist loadfile

# Group by class - runs all tests in a class on same worker
uv run pytest -n auto --dist loadscope
```

### GitHub Actions Configuration

#### Runner Specifications (as of 2024)

For public repositories using `ubuntu-latest`:
- **CPU cores**: 4 vCPUs
- **RAM**: 16 GB
- **Job timeout**: 360 minutes (6 hours) default
- **Workflow timeout**: 72 hours total

#### Modify Existing Workflow

**Important**: weathercop already has GitHub Actions workflows. Modify the existing test workflow rather than creating a new one.

Look for the existing workflow file in `.github/workflows/` (commonly named `tests.yml`, `ci.yml`, or `test.yml`).

**Key modifications to make**:

1. **Add timeout to the test job**:
   ```yaml
   jobs:
     test:
       timeout-minutes: 60  # Add this to existing job
   ```

2. **Update dependency installation** to use uv:
   ```yaml
   - name: Install dependencies
     run: |
       uv sync --dev
   ```

3. **Split test execution into two phases**:
   ```yaml
   - name: Run CPU-intensive tests (parallel)
     timeout-minutes: 30
     run: uv run pytest -n auto -m "not memory_intensive" --durations=10
   
   - name: Run memory-intensive tests (sequential)
     timeout-minutes: 30
     run: uv run pytest -n 1 -m "memory_intensive" --durations=10
   ```

#### Example of Modified Test Steps

Here's what the test steps section might look like after modification:

```yaml
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'  # Match your project's Python version
    
    - name: Install uv
      uses: astral-sh/setup-uv@v3
    
    - name: Install dependencies
      run: uv sync --dev
    
    - name: Run CPU-intensive tests (parallel)
      timeout-minutes: 30
      run: uv run pytest -n auto -m "not memory_intensive" --durations=10
    
    - name: Run memory-intensive tests (sequential)
      timeout-minutes: 30
      run: uv run pytest -n 1 -m "memory_intensive" --durations=10
```

#### Conservative Alternative (if memory issues persist)

With 16 GB RAM, you might be able to run memory-intensive tests with limited parallelization:

```yaml
    - name: Run memory-intensive tests (limited parallel)
      timeout-minutes: 30
      run: uv run pytest -n 2 -m "memory_intensive" --durations=10
```

### Testing the Configuration

1. **Mark the tests** in your test files
2. **Test locally** to verify markers work:
   ```bash
   uv run pytest --markers  # Should show memory_intensive marker
   uv run pytest -m "memory_intensive" --collect-only  # Shows which tests are marked
   uv run pytest -m "not memory_intensive" --collect-only  # Shows unmarked tests
   ```
3. **Run phased execution locally**:
   ```bash
   uv run pytest -n auto -m "not memory_intensive" -v
   uv run pytest -n 1 -m "memory_intensive" -v
   ```
4. **Monitor resource usage** to tune worker count for memory-intensive tests

### Expected Performance Improvements

- **test_copulae.py**: Full parallelization across 4 cores (locally may be more)
- **test_ensemble.py** and **test_multisite.py**: Sequential execution prevents OOM
- **Other tests**: Full parallelization by default
- **Overall**: Significant speedup while maintaining stability

### Troubleshooting

**If tests still fail with OOM:**
- Reduce parallelization further: `-n 2` or `-n 1`
- Mark additional tests as `memory_intensive`
- Use `--dist loadfile` to keep test files isolated: `uv run pytest -n auto --dist loadfile`

**If tests timeout:**
- Increase `timeout-minutes` for specific steps
- Investigate slow tests with `uv run pytest --durations=20`
- Consider splitting very slow tests into separate jobs

**If parallelization doesn't help:**
- Check test independence (shared state, fixtures, etc.)
- Verify no race conditions in parameterized tests
- Use `uv run pytest -n auto --dist loadfile` to run files sequentially

### Additional Optimization Tips

1. **Cache uv dependencies** in GitHub Actions (if not already present):
   ```yaml
   - name: Enable caching
     uses: astral-sh/setup-uv@v3
     with:
       enable-cache: true
   ```

2. **Use pytest's built-in parallelization** for parameterized tests within the same worker

3. **Profile test execution**:
   ```bash
   uv run pytest --durations=20  # Show 20 slowest tests
   ```

4. **Consider splitting by test type** into separate GitHub Actions jobs if needed

## Next Steps

1. Add `pytest-xdist` to project dev dependencies with `uv add --dev pytest-xdist`
2. Mark `test_ensemble.py` and `test_multisite.py` with `memory_intensive`
3. Register the marker in pytest configuration
4. Test locally with phased execution
5. Modify existing GitHub Actions workflow to use phased test execution
6. Monitor initial runs and adjust worker counts as needed

## References

- pytest-xdist documentation: https://pytest-xdist.readthedocs.io/
- GitHub Actions documentation: https://docs.github.com/en/actions
- pytest markers: https://docs.pytest.org/en/stable/how-to/mark.html
