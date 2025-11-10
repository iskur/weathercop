# Threading Migration Design: simulate_ensemble

**Date:** 2025-11-10
**Author:** Collaborative design with Claude Code
**Status:** Designed, ready for implementation
**Motivation:** Reduce memory overhead from multiprocessing + enable future free-threaded Python

---

## Overview

Replace `multiprocessing.Pool` with `concurrent.futures.ThreadPoolExecutor` for ensemble generation. Use thread-local VG copies and thread-local RNGs to ensure deterministic, race-condition-free parallel simulation.

**Key wins:**
- Memory: ~1-2MB per thread vs ~500MB per process (~250x reduction with 8 workers)
- Cleaner state: no locking needed during simulation
- Foundation for free-threaded Python migration

---

## Architecture

### Current Approach (Multiprocessing)
```
Main Process
├── Multisite object
├── Pool of N worker processes
└── Each worker:
    - Full copy of Multisite via pickle (~500MB)
    - Isolated OS process
    - No shared state
    - Safe but expensive
```

### Proposed Approach (Threading)
```
Main Thread
├── Multisite object (shared read-only)
├── wcop.vgs (shared, but thread-local copies created at startup)
└── ThreadPoolExecutor with N worker threads
    └── Each worker thread:
        - Reference to Multisite (read-only)
        - Deep copy of wcop.vgs in thread-local storage
        - Thread-local numpy.random.Generator
        - No locks needed during simulate()
```

**Why this works:**
1. xarray data is created fresh per-worker (no sharing)
2. VG state modifications are isolated to each thread
3. RNG sequences are deterministic but thread-isolated
4. File I/O is per-realization (no conflicts)

---

## Component: VARWG Thread-Local RNG

**Problem:** `varwg.rng` is a global object. Multiple threads calling `varwg.rng.normal()` or `varwg.reseed()` causes race conditions and interleaved RNG state.

**Solution:** Implement thread-local RNG in VARWG.

### Changes to `varwg/__init__.py`

```python
from numpy.random import default_rng
import threading

_thread_rng = threading.local()

# Keep old rng for backwards compatibility (deprecated)
rng = default_rng()

def get_rng():
    """Get thread-local RNG generator.

    Each thread gets its own independent Generator instance.
    First call in a thread creates a new Generator; subsequent
    calls return the same one.

    Returns
    -------
    numpy.random.Generator
        Thread-local random number generator
    """
    if not hasattr(_thread_rng, 'rng'):
        _thread_rng.rng = default_rng()
    return _thread_rng.rng

def reseed(seed):
    """Seed the thread-local RNG.

    Only affects the RNG for the current thread.

    Parameters
    ----------
    seed : int
        Seed value for the thread's RNG
    """
    rng_instance = get_rng()
    BitGen = type(rng_instance.bit_generator)
    rng_instance.bit_generator.state = BitGen(seed).state
```

### Call Site Updates

Replace all instances of `varwg.rng` with `varwg.get_rng()`:

**In varwg/core/core.py:**
- Line 223, 234: `varwg.rng.random()` → `varwg.get_rng().random()`
- Line 711, 713: `varwg.rng.bit_generator.state` → `varwg.get_rng().bit_generator.state`
- Line 1920, 2265, 2267: Similar replacements

**In varwg/time_series_analysis/ modules:**
- models.py: ~20 replacements
- phase_randomization.py: ~5 replacements
- rain_stats.py: ~3 replacements

**Total:** ~80 call sites across VARWG codebase.

### Verification

```python
# Each thread gets deterministic sequence
def worker(seed):
    varwg.reseed(seed)
    return [varwg.get_rng().normal() for _ in range(3)]

thread1_values = worker(42)  # [v1, v2, v3]
thread2_values = worker(42)  # Same [v1, v2, v3]
# Both threads seeded identically produce identical sequences
```

---

## Component: WeatherCop Thread-Local VGs

**Problem:** `simulate()` accesses `wcop.vgs` (shared). With multiple threads, modifications to VG state (primary_var, climate_signal, disturbance_std, etc.) cause race conditions.

**Solution:** Each thread gets a deep copy of `wcop.vgs` in thread-local storage.

### Changes to `weathercop/multisite.py`

**Add at module level (after imports):**

```python
import threading
import copy

_thread_local = threading.local()

def _worker_initializer(wcop_vgs):
    """Called once per worker thread at startup.

    Creates thread-local copy of VG objects so each thread
    can modify state independently.
    """
    _thread_local.vgs = copy.deepcopy(wcop_vgs)
```

**Modify `simulate()` signature:**

```python
def simulate(
    wcop,
    sim_times,
    *args,
    vgs=None,  # NEW: thread-local VGs passed in
    phase_randomize=True,
    ...
):
    """Simulate weather ensemble.

    Parameters
    ----------
    vgs : dict or None, optional
        VG objects to use. If None, uses wcop.vgs (for backwards compat).
        When called from threaded workers, pass thread-local vgs.
    """
    # Use passed-in vgs if available, otherwise fall back to wcop.vgs
    if vgs is None:
        vgs = wcop.vgs

    # Rest of function unchanged - use local 'vgs' variable
    # instead of wcop.vgs
    ...
```

**Modify `sim_one()` to use thread-local VGs:**

```python
def sim_one(args):
    (
        real_i,
        total,
        wcop,
        filepath,
        ...
    ) = args

    # Seed this thread's RNG
    varwg.reseed((1000 * real_i))

    # Load phases (file I/O, before simulation)
    if filepath_rphases_src:
        rphases = np.load(filepath_rphases_src.with_suffix(".npy"))
    else:
        rphases = None

    # Use thread-local VGs (no lock needed)
    return_trans = filepath_trans is not None
    sim_result = simulate(
        wcop,
        sim_times,
        rphases=rphases,
        vgs=_thread_local.vgs,  # NEW: pass thread-local VGs
        *sim_args,
        **sim_kwds,
    )

    # Rest unchanged: file I/O, disaggregation, etc.
    ...
```

---

## Component: Replace Pool with ThreadPoolExecutor

**Current code (lines 1713-1745):**

```python
from multiprocessing import Pool, Lock, current_process

lock = Lock()  # REMOVE THIS

with Pool(cop_conf.n_nodes) as pool:
    completed_reals = list(
        tqdm(
            pool.imap(
                sim_one,
                zip(realizations, repeat(...), ...),
                chunksize=max(1, n_realizations // cop_conf.n_nodes)
            ),
            total=len(realizations) + 1,
            initial=1,
        )
    )
```

**Proposed code:**

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(
    max_workers=cop_conf.n_nodes,
    initializer=_worker_initializer,
    initargs=(self.vgs,)
) as executor:
    completed_reals = list(
        tqdm(
            executor.map(
                sim_one,
                zip(realizations, repeat(...), ...),
                timeout=None  # Some tasks can take minutes
            ),
            total=len(realizations)
        )
    )
```

**Changes:**
- Remove `from multiprocessing import Pool, Lock, current_process` (keep other imports)
- Add `from concurrent.futures import ThreadPoolExecutor`
- Add `_worker_initializer` function
- Replace `Pool()` context manager with `ThreadPoolExecutor()`
- Pass `initializer` and `initargs` to ThreadPoolExecutor
- Remove `chunksize` (ThreadPoolExecutor doesn't use it)
- Update `tqdm` arguments (adjust `total` and `initial`)

### No Lock Needed

The current `with lock:` context at lines 324-335 in `simulate()` can be **completely removed** because:
- Thread-local VGs are not shared
- Thread-local RNG has no contention
- GIL protects individual Python operations for metadata reads

---

## Data Flow: Single Realization

```
Main Thread: simulate_ensemble(n_realizations=100)
│
├─ First realization (sequential, single-threaded)
│  ├─ varwg.reseed(0)
│  ├─ simulate(wcop, sim_times) using wcop.vgs directly
│  └─ Write to disk
│
├─ ThreadPoolExecutor created with initializer
│  │  (initializer runs 8 times, creates _thread_local.vgs in each thread)
│  │
│  ├─ Thread 1: sim_one(real_i=1)
│  │  ├─ varwg.reseed(1000) ← affects only Thread 1's RNG
│  │  ├─ simulate(..., vgs=_thread_local.vgs)
│  │  │  └─ All varwg.get_rng() calls use Thread 1's RNG
│  │  └─ Write to disk
│  │
│  ├─ Thread 2: sim_one(real_i=2)
│  │  ├─ varwg.reseed(2000) ← affects only Thread 2's RNG
│  │  ├─ simulate(..., vgs=_thread_local.vgs)
│  │  └─ Write to disk
│  │
│  └─ ... (Threads 3-8 similarly)
│
└─ Load ensemble from disk files
```

---

## Memory Comparison

### Multiprocessing (Current)
```
Main process: 500MB (Multisite object)
Worker 1: 500MB (pickled copy)
Worker 2: 500MB (pickled copy)
...
Worker 8: 500MB (pickled copy)
────────────────
Total: ~4GB
```

### Threading (Proposed)
```
Main thread: 500MB (Multisite object)
  + Thread-local vgs copy 1: 10MB
  + Thread-local vgs copy 2: 10MB
  + Thread-local RNG 1: 0.1MB
  + Thread-local RNG 2: 0.1MB
  ...
────────────────
Total: ~600MB (parent + 8 threads)
```

**Savings:** ~85% memory reduction for 8 workers.

---

## Determinism and Reproducibility

**Guarantee:** Same `real_i` always produces identical realization.

**Mechanism:**
1. `varwg.reseed((1000 * real_i))` is deterministic (same input → same seed)
2. Thread-local RNG with fixed seed produces fixed sequence
3. VG parameters are fixed (from initial fit)
4. Random phase generation uses seeded RNG

**Example:**
```python
# Realization 5 in run 1
varwg.reseed(5000)  # Thread 3's RNG
# ... simulation produces realization_5_run1.nc

# Realization 5 in run 2
varwg.reseed(5000)  # Thread 7's RNG (different thread, same seed)
# ... simulation produces realization_5_run2.nc
# Identical to realization_5_run1.nc (within floating-point precision)
```

---

## Error Handling and Edge Cases

### Thread Exception Handling
`ThreadPoolExecutor.map()` will raise the first worker exception to the main thread, stopping iteration. Current behavior (wait for all workers) is maintained.

### File I/O Conflicts
Each realization writes to its own file (`real_0000.nc`, `real_0001.nc`, etc.). No conflicts.

### Long-Running Simulations
Set `timeout=None` to allow simulations that exceed default timeout.

---

## Testing Strategy

### Unit Tests
1. **Thread-local RNG:**
   - Verify each thread gets independent sequence
   - Verify reseed is deterministic per thread
   - Verify 2+ threads with same seed produce identical sequences

2. **Thread-local VGs:**
   - Verify deep copy doesn't share state
   - Verify VG modifications in one thread don't affect others
   - Verify simulate() produces same results with thread-local vs shared VGs

### Integration Tests
1. **Ensemble generation:**
   - Run with threading, compare results to multiprocessing baseline
   - Check memory usage (should be ~85% lower)
   - Verify determinism (same n_realizations, same seeds → identical files)

2. **Stress tests:**
   - Run with 16+ threads to expose race conditions (should have none)
   - Run with thread count > CPU count (oversubscription)

### Regression Tests
1. Non-threaded code paths (legacy API) still work
2. `simulate()` with `vgs=None` still works
3. Single-threaded mode (n_nodes=1) works

---

## Implementation Phases

### Phase 1: VARWG (Thread-Local RNG)
- Add `get_rng()` and thread-local storage
- Replace 80 call sites
- Test for regressions

### Phase 2: WeatherCop (Thread-Local VGs + ThreadPoolExecutor)
- Add `_worker_initializer()` and thread-local storage
- Modify `simulate()` signature to accept `vgs` parameter
- Modify `sim_one()` to pass thread-local vgs
- Replace Pool with ThreadPoolExecutor
- Remove lock from simulate()

### Phase 3: Testing
- Unit tests for thread-local RNG
- Integration tests for ensemble generation
- Regression tests for non-threaded code paths

### Phase 4: Cleanup
- Remove deprecated `varwg.rng` (if not needed for backwards compat)
- Update documentation
- Remove multiprocessing imports if unused elsewhere

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Forgot to update a `varwg.rng` call site | Use grep/ripgrep to find remaining calls; systematic replacement |
| Deep copy of VGs is expensive | Measure actual time; only optimization if needed |
| ThreadPoolExecutor behaves differently than Pool | Use explicit `executor.map()` with same semantics; test throughput |
| Backwards compatibility broken | Keep `varwg.rng` functional; support `vgs=None` in simulate() |
| Race condition in VG initialization | Use initializer callback (runs once per thread at startup) |

---

## Success Criteria

1. ✅ Ensemble generation produces identical results (floating-point precision)
2. ✅ Memory usage ~85% lower (monitoring with memory_diagnostics.py)
3. ✅ Execution time comparable or faster (threading overhead minimal)
4. ✅ All existing tests pass (regression)
5. ✅ New threading tests pass (unit + integration)
6. ✅ Determinism verified (same seed → same realization, different runs)
