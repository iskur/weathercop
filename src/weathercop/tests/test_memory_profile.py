"""Memory profiling for large tests - optional but useful for monitoring."""
import pytest
import tracemalloc
from weathercop.multisite import Multisite


@pytest.mark.skip(reason="Manual profiling - run with: pytest -v -s -k memory_profile")
def test_ensemble_memory_profile(multisite_instance):
    """Profile memory usage during ensemble generation (manual test)."""
    tracemalloc.start()

    # Simulate small ensemble
    multisite_instance.simulate_ensemble(
        n_realizations=2,
        name="memory_test",
        clear_cache=True,
        write_to_disk=False,
    )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Log memory usage (not an assertion, just for manual review)
    print(f"\nMemory - Current: {current / 1024 / 1024:.1f} MB")
    print(f"Memory - Peak: {peak / 1024 / 1024:.1f} MB")

    # Very loose check - if peak is > 4GB on a small 2-realization ensemble,
    # something is wrong
    assert peak < 4 * 1024 * 1024 * 1024, (
        f"Ensemble simulation used excessive memory: {peak / 1024 / 1024 / 1024:.1f} GB"
    )
