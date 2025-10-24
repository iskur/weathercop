# Memory Diagnostics Guide

## Overview

WeatherCop includes comprehensive memory diagnostics to track OOM issues during test execution. Diagnostics run automatically and log to `~/.weathercop/test_memory.log`.

## How It Works

Three layers of monitoring:

1. **System-level**: `/proc/self/status` snapshots (VmRSS, VmPeak, VmSize)
2. **Python-level**: `tracemalloc` peak memory per test
3. **Application-level**: xarray dataset tracking and garbage collection stats

## Accessing Logs

During or after test execution:

```bash
# Watch log in real-time
tail -f ~/.weathercop/test_memory.log

# After test completes, analyze results
python scripts/analyze_memory_log.py ~/.weathercop/test_memory.log
```

## Disabling Diagnostics

For performance-critical runs:

```bash
export WEATHERCOP_DISABLE_DIAGNOSTICS=1
pytest src/weathercop/tests/
```

## Understanding Log Format

Each test generates entries like:

```
[2025-10-24T12:34:56.789] TEST_START: test_name
  Elapsed: 12.3s
  /proc/self/status:
    VmRSS: 2048000 kB
    VmPeak: 2560000 kB
  Top tracemalloc allocations:
    script.py:100: 512.3 MB
  GC: collections=42, uncollectable=0

[2025-10-24T12:35:00.123] TEST_END: test_name
  Elapsed: 16.5s
  /proc/self/status:
    VmRSS: 512000 kB
    VmPeak: 2560000 kB
  Top tracemalloc allocations:
    (tracemalloc not active)
  Peak memory: 2500.0 MB
  GC: collections=45, uncollectable=0
  Open xarray datasets: 0
```

## Post-Mortem Analysis After OOM

When Emacs gets killed by OOM:

1. Check if pytest is still running (might be)
2. Analyze the log:
   ```bash
   python scripts/analyze_memory_log.py ~/.weathercop/test_memory.log
   ```
3. Look for:
   - Last TEST_END entry (test that triggered OOM)
   - Largest peak_memory_mb values
   - Growing xarray_count (indicates leaks)
   - Sudden memory spikes

## Common Patterns

### Gradual Memory Growth
Tests accumulate memory over time without cleanup:
- Check fixture scoping (should be function-scoped)
- Look for unclosed xarray datasets
- Verify matplotlib figures are closed with `plt.close('all')`

### Sudden Spike
A single test uses >1GB:
- Reduce ensemble test sizes
- Use chunking in xarray
- Disable parallel xarray loading (`WEATHERCOP_PARALLEL_LOADING=0`)

### High xarray_count
Datasets not being closed:
- Check conftest.py cleanup code
- Verify `ds.close()` is called
- Look at multisite_instance fixture cleanup

## Debugging Specific Tests

Run a single test with diagnostics enabled and memory in focus:

```bash
# Watch memory log while test runs
tail -f ~/.weathercop/test_memory.log &
pytest src/weathercop/tests/test_multisite.py::Test::test_name -v --tb=short

# After test, check peak memory
python scripts/analyze_memory_log.py ~/.weathercop/test_memory.log
```

## Configuration

- **Log location**: `WEATHERCOP_DIR/test_memory.log` (default: `~/.weathercop/test_memory.log`)
- **Disable diagnostics**: `WEATHERCOP_DISABLE_DIAGNOSTICS=1`

## Implementation Details

- **Unbuffered logging**: Lines are flushed immediately with `os.fsync()`
- **Survives OOM kills**: Pytest continues running even if Emacs/parent is killed
- **Weakref tracking**: xarray tracker doesn't hold references to datasets
- **tracemalloc overhead**: ~5% performance impact
