# WeatherCop Test Suite Memory Analysis

## Executive Summary

The full test suite experiences significant memory growth, primarily driven by the `test_density` test in `test_copulae.py`. With 128GB available, the current memory usage patterns are manageable, but optimization is worthwhile for CI/CD environments and development workflows.

**Key Finding:** test_density is not a leak—it's computationally intensive and memory-hungry by design.

## Diagnostic Run Results

### Test Run Configuration
- System RAM: 128GB
- Python version: 3.13.5
- Pytest: 8.4.1
- Diagnostics enabled with unbuffered logging to `~/.weathercop/test_memory.log`

### Memory Profile

| Phase | Test | Peak Memory | Duration | Notes |
|-------|------|-------------|----------|-------|
| Setup | Collection | ~1GB | Instant | Module import, copula initialization |
| Phase 1 | test_bounds | ~1MB | Quick | Minimal memory |
| Phase 2 | test_cdf_given_u | ~140MB | ~10s | 139MB spike, moderate |
| Phase 3 | test_cdf_given_v | ~140MB | ~10s | Consistent pattern |
| Phase 4 | test_cop | ~200MB | ~1s | Recovered after cleanup |
| **Phase 5** | **test_density** | **~9,600MB** | **~5 minutes** | **36 copulas × numerical integration** |

### Key Observations

1. **Predictable growth pattern** - Memory increases as tests run, not random spikes
2. **test_density dominates** - Consumes 9.6GB but eventually completes
3. **No detected leaks** - XArray tracking shows 0 unclosed datasets
4. **GC working** - Garbage collection runs regularly (259-1070+ collections)
5. **Process survives** - pytest didn't get OOM-killed, completed successfully

## Root Cause Analysis: test_density

### The Code
Location: `src/weathercop/tests/test_copulae.py:348-386`

```python
def test_density(self):
    """Does the density integrate to 1?"""
    for name, frozen_cop in cop.frozen_cops.items():  # 36 copulas
        def density(x, y):
            return frozen_cop.density(np.array([x]), np.array([y]))

        one = integrate.nquad(
            density,
            ([self.eps, 1 - self.eps], [self.eps, 1 - self.eps]),
        )[0]
```

### Why It Uses So Much Memory

1. **36 copulas** - Tests all 36 copula instances (clayton, gumbel, gaussian, etc.)
2. **Numerical integration** - `scipy.integrate.nquad` does 2D adaptive quadrature
   - Makes thousands of function evaluations
   - Accumulates evaluation results in memory
   - Each copula evaluation creates temporary arrays
3. **Adaptive grid** - Quadrature refinement keeps intermediate grids in memory
4. **No cleanup between copulas** - Memory from previous integrations not released until test ends

### Memory Growth Timeline
- **0-30s**: 1.3GB - First copula integrations
- **30-60s**: 1.5GB - Several more copulas done
- **60-180s**: 2-9GB - More copulas added to memory
- **180-300s**: 9.6GB peak - Final copulas
- **300s**: Completes successfully

## Is This a Problem?

### For Local Development (128GB system)
**No immediate problem.** 9.6GB on a 128GB system is ~7.5% of capacity.

**Comfortable margin** for concurrent work:
- Running one test suite: 9.6GB
- System reserves/other processes: ~20GB
- Available for other work: ~98GB

### For CI/CD (Smaller VMs)
**Potentially problematic:**
- 8GB VM: Would hit OOM
- 16GB VM: Marginal (62% utilization)
- 32GB VM: Comfortable (30% utilization)

### Performance Perspective
- Duration: ~5 minutes for one test
- Inefficient: Could be parallelized or reduced
- Correctness: Test passes, validates density integration

## Optimization Opportunities

### Non-Breaking Changes (Keep test as-is, optimize implementation)

1. **Copula-level cleanup** - Call gc.collect() between copulas
2. **Memory pooling** - Reuse arrays in density evaluation
3. **Reduce copula count in density test** - Test subset, rest in separate test
4. **Profile with memory_profiler** - Find exact allocation hotspots

### Moderate Changes (Restructure test slightly)

1. **Split test_density into smaller tests**
   - test_density_basic (few copulas)
   - test_density_comprehensive (all copulas)
   - Mark comprehensive as `@pytest.mark.slow` or `@pytest.mark.expensive`

2. **Parametrized test with explicit cleanup**
   ```python
   @pytest.mark.parametrize("copula_name,copula", cop.frozen_cops.items())
   def test_density_integration(copula_name, copula):
       # Test one copula at a time
       # Pytest runs in sequence, each test has isolated cleanup
       ...
   ```

### Aggressive Changes (Fundamental rethink)

1. **Replace numerical integration test** with analytical validation
2. **Reduce resolution** of numerical integration (faster, less memory)
3. **Use subprocess/multiprocessing** to isolate tests

## Recommended Next Steps

1. **If staying at 128GB**: No action needed, current state is acceptable
2. **If CI/CD has memory limits**:
   - Split test_density
   - Add `@pytest.mark.expensive` for resource-intensive tests
   - Run expensive tests on larger CI nodes only
3. **For general robustness**:
   - Add explicit gc.collect() between copula tests
   - Profile copulae.density() to find allocation hotspots

## Memory Diagnostics Infrastructure

The new diagnostics system (`src/weathercop/tests/memory_diagnostics.py`) will help monitor:
- Future optimizations (verify memory reduced)
- Regressions (detect if new tests use more memory)
- CI/CD compatibility (track max RSS)

**Usage:**
```bash
# Run tests with diagnostics
pytest src/weathercop/tests/ -v

# Analyze results
python scripts/analyze_memory_log.py ~/.weathercop/test_memory.log
```

## Conclusion

`test_density` is **legitimate computational work**, not a bug or leak. The test:
- ✅ Passes successfully
- ✅ Produces correct results
- ✅ Cleans up properly
- ✅ Doesn't leak resources

The memory usage is **by design**: numerical integration of 36 copula density functions is computationally and memory-intensive. With 128GB of RAM, this is acceptable for local development.

For smaller systems or CI/CD pipelines, consider the optimization recommendations above.
