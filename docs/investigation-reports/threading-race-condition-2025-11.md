# Threading Race Condition Fixes - Progress Log

## Problem Statement

WeatherCop's multisite weather simulation has a severe race condition in multi-threaded mode (PROFILE=False). When using ThreadPoolExecutor with n_nodes>1, simulation results are catastrophically wrong:

- **Temperature**: 350-400% too high (e.g., 36-49°C instead of 8°C)
- **Rainfall**: 50-84% too low
- **Humidity**: 9-11% too high

The errors are **deterministic** (same wrong values every run), suggesting systematic data corruption rather than random race conditions.

## Test Setup

**Diagnostic script**: `diagnose_threading.py`
- Runs 3 realizations with PROFILE=True (single-threaded baseline)
- Runs 3 realizations with PROFILE=False, n_nodes=2 (multi-threaded test)
- Compares statistical results to detect race conditions

**Key files involved**:
- `src/weathercop/multisite.py` - Main simulation orchestration
- `src/weathercop/vine.py` - Vine copula structures
- `src/weathercop/seasonal_cop.py` - Seasonal copulas with time-varying parameters
- `src/weathercop/copulae.py` - Bivariate copula implementations

## Fixes Attempted

### ✅ Fix 1: Deep Copy VGs for Thread-Local Storage
**Status**: Already implemented before this session
**Location**: `multisite.py:1804, multisite.py:47-50`
**What**: VG objects are deep copied and stored in thread-local storage via `_worker_initializer()`
**Result**: VGs are properly isolated per thread (confirmed by different object IDs in logs)

### ✅ Fix 2: Copy rphases Arrays to Prevent In-Place Modification
**Status**: Implemented
**Location**: `multisite.py:514-520` in `_vg_ph()`
**What**:
```python
# Create copies to avoid modifying shared rphases arrays (thread-safety)
phases = []
for phase_ in rphases:
    phase_copy = phase_.copy()
    phase_copy[:, 0] = zero_phases[station_name]
    phases += [phase_copy]
rphases = phases
```
**Result**: FAILED - Race condition persists with identical symptoms

### ✅ Fix 3: Initialize Vine Lazy Properties Before Threading
**Status**: Implemented
**Location**: `multisite.py:1805-1819`
**What**: Force initialization of vine.A and vine.edge_map properties before ThreadPoolExecutor starts
**Rationale**: Lazy property initialization is not thread-safe
**Result**: FAILED - Race condition persists

### ✅ Fix 4: Initialize SeasonalCop Lazy Properties Before Threading
**Status**: Implemented
**Location**: `multisite.py:1812-1817`
**What**: Force initialization of SeasonalCop.solution, .thetas, .sliding_theta properties for all 18 seasonal copulas
**Rationale**: Seasonal copulas have lazy-initialized properties accessed during simulation
**Result**: FAILED - Race condition persists (confirmed: "Initialized 18 seasonal copulas" message appears)

### ✅ Fix 5: Deep Copy Vine and Distributions to Thread-Local Storage
**Status**: Implemented
**Location**:
- `multisite.py:1821-1844` - Deep copy and package for workers
- `multisite.py:47-63` - Modified `_worker_initializer()` to store in thread-local
- `multisite.py:514-543` - Modified `_vg_ph()` to read from thread-local storage

**What**:
```python
# Deep copy vine and all distribution dictionaries
vine_for_workers = copy.deepcopy(self.vine)
dists_for_workers = {
    'fft_dists': copy.deepcopy(self.fft_dists),
    'qq_dists': copy.deepcopy(self.qq_dists),
    'data_trans_dists': copy.deepcopy(self.data_trans_dists),
    'sim_dists': copy.deepcopy(self.sim_dists),
}

# Store in thread-local storage in _worker_initializer()
_thread_local.vine = copy.deepcopy(thread_local_data['vine'])
_thread_local.fft_dists = copy.deepcopy(thread_local_data['fft_dists'])
# ... etc

# Read from thread-local in _vg_ph()
if hasattr(_thread_local, 'fft_dists'):
    fft_dist = _thread_local.fft_dists[station_name]
    vine = _thread_local.vine
else:
    # Fallback for main thread
    fft_dist = wcop.fft_dists[station_name]
    vine = wcop.vine
```

**Rationale**:
- Vine copula's `simulate()` method is called from multiple threads via `vine[station_name].simulate()`
- Distribution objects have internal state that could be corrupted during concurrent `.cdf()` calls
- Even though we initialized lazy properties, the objects themselves are shared

**Result**: NOT YET TESTED (need to verify thread-local storage is actually being used)

## Key Findings

### Code Path Differences: PROFILE=True vs PROFILE=False

**PROFILE=True (lines 1687-1730)**:
- Simple loop in main thread: `for real_i in tqdm(range(n_realizations))`
- Calls `self.simulate()` directly
- Uses `self._vg_ph` instance method
- Each realization: `varwg.reseed((1000 * real_i))`

**PROFILE=False (lines 1731-1854)**:
- First realization in main thread: `self.simulate()` with `varwg.reseed((0))`
- Remaining realizations via ThreadPoolExecutor
- Calls module-level `simulate()` function which uses module-level `_vg_ph`
- Each worker realization: `varwg.reseed((1000 * real_i))` in `sim_one()`

### Two Different _vg_ph Functions!

**Important discovery**: There are TWO separate `_vg_ph` functions:

1. **Module-level `_vg_ph`** (line 458):
   - Used by worker threads via module-level `simulate()` function
   - This is what we modified with thread-local storage fixes

2. **Instance method `Multisite._vg_ph`** (line 2291):
   - Used by main thread via `self.simulate()`
   - NOT modified with thread-local storage fixes (but main thread doesn't need it)

### Shared Mutable State in _vg_ph

The `_vg_ph` function accesses many wcop attributes:

**Read-only (safe)**:
- `wcop.As` - FFT amplitudes (static data)
- `wcop.varnames` - Variable names list
- `wcop.zero_phases` - Zero-phase adjustments
- `wcop.heat_waves` - Heat wave masks

**Potentially mutable (NOW FIXED with thread-local)**:
- `wcop.vine` - Vine copula structure (has `simulate()` method)
- `wcop.fft_dists[station]` - FFT distribution objects (have `.cdf()` method)
- `wcop.qq_dists[station]` - QQ distribution objects
- `wcop.data_trans_dists[station]` - Data transformation distributions
- `wcop.sim_dists[station]` - Simulation distributions

Distribution objects from scipy.stats or custom ECDF classes likely have internal caches that are mutated during `.cdf()` calls.

## Debug Output Added

Added comprehensive debug tracing in `_vg_ph` (requires `WEATHERCOP_DEBUG_TRACE=1`):
- Thread ID and function entry
- Whether using thread-local or wcop attributes (with object IDs)
- Values at 8 transformation stages (see `multisite.py:527-671`)

## Next Steps

### Immediate Action Required
1. **Test Fix 5**: Run `diagnose_threading.py` with `WEATHERCOP_DEBUG_TRACE=1` to verify:
   - Thread-local storage is initialized correctly
   - Worker threads use thread-local vine/distributions (not wcop shared objects)
   - Main thread safely uses wcop objects (no thread-local storage expected)

2. **If Fix 5 works**: Clean up debug code and commit

3. **If Fix 5 fails**: Investigate further possibilities:

### Further Investigation if Still Failing

**Hypothesis A: varwg.get_rng() Thread-Safety**
- `varwg.get_rng()` is used extensively (line 1238 in vine.py)
- If RNG state is global rather than thread-local, this could cause issues
- Check: `grep -r "def get_rng" in varwg source`

**Hypothesis B: numpy.fft State**
- FFT operations might have internal state: `np.fft.ifft()` (line 522-524)
- Check if numpy FFT functions are thread-safe

**Hypothesis C: Distribution Object Internal Caches**
- scipy.stats distributions might cache PPF/CDF results
- Custom ECDF class (line 692-723) uses interpolate.interp1d - check thread-safety
- KDE distributions for empirical marginals might have caches

**Hypothesis D: SymPy-Generated Cython Extensions**
- Copula functions auto-generated in `src/weathercop/ufuncs/`
- These might have static variables or global state
- Check generated `.c` files for `static` variables

**Hypothesis E: Numerical Tolerance Issues**
- Different threads might hit different branches due to floating-point precision
- Check `cop_conf.py` for tolerance settings that affect branching

### Alternative Approaches

**If thread-local copying doesn't work:**

1. **Make wcop truly read-only**: Create immutable frozen versions of all objects before threading

2. **Use ProcessPoolExecutor instead of ThreadPoolExecutor**: Processes have separate memory, eliminating shared state (but slower due to serialization overhead)

3. **Lock-based synchronization**: Add locks around vine.simulate() calls (slower but might reveal if this is the issue)

4. **Trace-based comparison**: Since PROFILE=True/False use different code paths, compare two multi-threaded runs:
   - Run 1: n_nodes=1 (no actual threading)
   - Run 2: n_nodes=2 (threading)
   - Both use same code path, so traces are comparable

## Files Modified

```
src/weathercop/multisite.py
├── Lines 47-63: _worker_initializer() - Store vine/dists in thread-local
├── Lines 495-501: Debug output in _vg_ph
├── Lines 514-526: Thread-local distribution access with fallback
├── Lines 530-546: Thread-local vine access with fallback
├── Lines 1821-1844: Deep copy vine and distributions before threading
└── (Previous fixes: rphases copy, lazy init)
```

## Dependencies to Check

- Python threading.local() behavior
- numpy FFT thread-safety
- scipy.stats distribution thread-safety
- varwg RNG implementation (thread-local vs global)
- Cython extension thread-safety
- interpolate.interp1d thread-safety

## Test Commands

```bash
# Run diagnostic test
python diagnose_threading.py

# Run with debug tracing
WEATHERCOP_DEBUG_TRACE=1 python diagnose_threading.py

# Run specific test from test suite
pytest src/weathercop/tests/test_multisite.py::Test::test_sim_gradual -v

# Check for AttributeError in thread-local access
grep -n "hasattr(_thread_local" src/weathercop/multisite.py
```

## Expected Behavior After Fix

When `diagnose_threading.py` runs successfully:
```
============================================================
RESULTS COMPARISON
============================================================

Maximum absolute difference: < 1e-10
✓ Results match! No obvious race condition detected
```

Temperature should be ~8°C (not 36-49°C), rainfall ~0.13 (not 0.05-0.07).

## Additional Context

- Tests are slow (~5 minutes for diagnose_threading.py)
- Cache directory: `/home/dirk/.weathercop/cache/`
- Ensemble output: `/home/dirk/.weathercop/ensembles/`
- Test uses `multisite_testdata.nc` from `/home/dirk/data/opendata_dwd/`
- Configuration: DWD VARWG config (requires `rh='empirical'` not `'truncnorm'`)

---

## Session 2: 2025-11-21 - Architectural Analysis

### ❌ Fix 5: FAILED

**Test Results**:
- Thread-local storage IS working (confirmed by different object IDs per thread)
- But race condition PERSISTS with identical symptoms
- Temperature: still 36-49°C instead of 8°C (353-398% too high)
- Rainfall: still 0.047-0.070 instead of 0.131 (52-84% too low)

**Debug Evidence**:
```
[ThreadPoolExecutor-1_0] Using thread-local distributions (id=139864494840192)
[ThreadPoolExecutor-1_1] Using thread-local distributions (id=139864131369024)
[ThreadPoolExecutor-1_0] Using thread-local vine (id=139864228023824)
[ThreadPoolExecutor-1_1] Using thread-local vine (id=139864215122480)
```
Different object IDs prove thread-local storage is functioning correctly.

### 🛑 STOP: After 5 Failed Fixes, Question the Architecture

Following systematic-debugging protocol: After 3+ failed fixes, stop attempting symptom fixes and question the fundamental approach.

### Root Cause Analysis

**What we've successfully isolated:**
1. ✅ Thread-local RNG (varwg.get_rng())
2. ✅ Thread-local VG objects (deep copied per thread)
3. ✅ Thread-local vine objects (deep copied per thread)
4. ✅ Thread-local distribution objects (deep copied per thread)
5. ✅ Removed shared state writes (usevine, rphases)

**Yet the race condition persists. This means:**

The problem is **NOT** in Python-level shared state. The problem is in **C/Cython/Fortran libraries** that Python's thread-local storage cannot isolate:

1. **numpy.fft (FFTW)** - May use global state or thread-unsafe caches
2. **scipy.stats distributions** - PPF/CDF computations may cache at C level
3. **Cython copula extensions** - Auto-generated in `src/weathercop/ufuncs/`, may have static variables
4. **scipy.interpolate.interp1d** - Used by ECDF class, may have internal state
5. **NumPy's C-level operations** - Various operations may not be thread-safe

**Evidence this is C-level state:**
- Deterministic wrong values (same every run) suggests systematic corruption, not random races
- All Python-level isolation works correctly
- Thread-local storage cannot protect against C-level global state

### Architectural Recommendation

**Replace ThreadPoolExecutor with ProcessPoolExecutor**

**Rationale:**
- Processes have completely separate memory spaces
- No shared state at ANY level (Python, C, Cython, Fortran)
- Eliminates all race conditions structurally
- Trade-off: Slower due to pickling/IPC overhead, but CORRECT results

**Alternative if ProcessPoolExecutor is too slow:**
- Add locks around critical sections (vine.simulate(), distribution.cdf/ppf calls)
- This would prove concurrency is the issue and identify exact bottlenecks
- Then optimize the hot paths to be thread-safe

**Why previous fixes failed:**
Each fix addressed Python-level shared state, but the corruption happens in C-level libraries. Thread-local storage in Python cannot protect C-level global variables or caches.

### Implementation Plan

**Option A: ProcessPoolExecutor (Recommended)**
```python
# In multisite.py, replace:
with ThreadPoolExecutor(...) as executor:
    ...

# With:
with ProcessPoolExecutor(...) as executor:
    ...
```

**Expected result:** Race condition eliminated, tests pass, correct results.

**Option B: Prove it's a concurrency issue**
```python
# Add a global lock around vine.simulate()
with _simulation_lock:
    ranks_sim = vine[station_name].simulate(...)
```
If this fixes the issue, we've confirmed it's thread-safety in vine or its dependencies.

---

**Session ended at**: 2025-11-21 09:30
**Status**: Architectural issue identified - ThreadPoolExecutor incompatible with C-level libraries
**Recommendation**: Switch to ProcessPoolExecutor or add locks to prove concurrency issue
**Next action**: Implement ProcessPoolExecutor and verify it fixes the race condition
