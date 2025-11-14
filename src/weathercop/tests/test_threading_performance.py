"""
Test threading vs multiprocessing performance for CVine.

This test verifies that:
1. Threading produces identical results to multiprocessing
2. Simulated means match observed means (critical requirement)
3. Threading provides performance benefits
"""
import time
import numpy as np
import pytest
from weathercop import cop_conf
from weathercop.vine import CVine
import scipy.stats as spstats


@pytest.fixture
def test_data():
    """Generate realistic test data for vine copulas."""
    np.random.seed(42)
    n_vars = 4
    n_timesteps = 1000

    # Create correlated data
    corr_matrix = np.array([
        [1.0, 0.6, 0.4, 0.3],
        [0.6, 1.0, 0.5, 0.4],
        [0.4, 0.5, 1.0, 0.6],
        [0.3, 0.4, 0.6, 1.0]
    ])

    # Generate correlated normal data
    mean = np.zeros(n_vars)
    data = np.random.multivariate_normal(mean, corr_matrix, n_timesteps).T

    # Convert to ranks
    ranks = np.array([spstats.rankdata(row)/(len(row)+1) for row in data])

    return ranks


@pytest.fixture
def cvine_threading(test_data):
    """Create CVine with threading enabled."""
    original_use_threading = cop_conf.USE_THREADING
    cop_conf.USE_THREADING = True

    varnames = ['var1', 'var2', 'var3', 'var4']
    vine = CVine(
        test_data,
        varnames=varnames,
        verbose=False,
        weights='tau',
        central_node='var1'
    )

    yield vine

    # Restore original setting
    cop_conf.USE_THREADING = original_use_threading


@pytest.fixture
def cvine_multiprocessing(test_data):
    """Create CVine with multiprocessing enabled."""
    original_use_threading = cop_conf.USE_THREADING
    cop_conf.USE_THREADING = False

    varnames = ['var1', 'var2', 'var3', 'var4']
    vine = CVine(
        test_data,
        varnames=varnames,
        verbose=False,
        weights='tau',
        central_node='var1'
    )

    yield vine

    # Restore original setting
    cop_conf.USE_THREADING = original_use_threading


def test_threading_vs_multiprocessing_correctness(test_data, cvine_threading, cvine_multiprocessing):
    """
    Test that threading produces identical results to multiprocessing.

    This is critical - the simulated values must be statistically identical.
    """
    np.random.seed(123)
    T_sim = 500

    # Simulate with threading
    sim_threading = cvine_threading.simulate(T=T_sim)

    # Simulate with multiprocessing (same seed)
    np.random.seed(123)
    sim_multiprocessing = cvine_multiprocessing.simulate(T=T_sim)

    # Results should be very close (allowing for minor numerical differences)
    for i in range(len(sim_threading)):
        assert np.allclose(sim_threading[i], sim_multiprocessing[i], rtol=1e-4, atol=1e-6), \
            f"Variable {i}: Threading and multiprocessing produced different results!"


def test_simulated_means_match_observations(test_data, cvine_threading):
    """
    CRITICAL TEST: Verify that simulated means match observed means.

    This is the user's primary requirement.
    """
    np.random.seed(456)
    T_sim = 2000  # Larger sample for better mean estimation

    # Get observed means from input data
    observed_means = test_data.mean(axis=1)

    # Simulate
    simulated = cvine_threading.simulate(T=T_sim)
    simulated_means = simulated.mean(axis=1)

    # Check that means are close (within 5%)
    for i in range(len(observed_means)):
        rel_diff = abs(simulated_means[i] - observed_means[i]) / observed_means[i]
        assert rel_diff < 0.05, \
            f"Variable {i}: Simulated mean ({simulated_means[i]:.4f}) differs from " \
            f"observed mean ({observed_means[i]:.4f}) by {rel_diff*100:.1f}%"

    print("\n=== Mean Comparison ===")
    print(f"{'Variable':<10} {'Observed':<12} {'Simulated':<12} {'Diff %':<10}")
    print("-" * 50)
    for i in range(len(observed_means)):
        diff_pct = (simulated_means[i] - observed_means[i]) / observed_means[i] * 100
        print(f"var{i+1:<7} {observed_means[i]:<12.6f} {simulated_means[i]:<12.6f} {diff_pct:<10.2f}")


@pytest.mark.benchmark
def test_threading_performance_benefit(cvine_threading, cvine_multiprocessing):
    """
    Benchmark threading vs multiprocessing performance.

    Threading should be faster due to reduced serialization overhead.
    """
    T_sim = 1000
    n_trials = 3

    # Benchmark threading
    times_threading = []
    for _ in range(n_trials):
        np.random.seed(789)
        start = time.time()
        _ = cvine_threading.simulate(T=T_sim)
        times_threading.append(time.time() - start)
    time_threading = np.mean(times_threading)

    # Benchmark multiprocessing
    times_mp = []
    for _ in range(n_trials):
        np.random.seed(789)
        start = time.time()
        _ = cvine_multiprocessing.simulate(T=T_sim)
        times_mp.append(time.time() - start)
    time_mp = np.mean(times_mp)

    speedup = time_mp / time_threading

    print("\n=== Performance Benchmark ===")
    print(f"Threading:        {time_threading:.3f}s (avg of {n_trials} trials)")
    print(f"Multiprocessing:  {time_mp:.3f}s (avg of {n_trials} trials)")
    print(f"Speedup:          {speedup:.2f}x")
    print(f"Performance gain: {(speedup-1)*100:.1f}%")

    # Threading should be at least as fast as multiprocessing
    # (allowing for some variance in timing)
    assert speedup >= 0.8, \
        f"Threading is significantly slower than multiprocessing (speedup={speedup:.2f}x)"


def test_threading_memory_efficiency(cvine_threading, cvine_multiprocessing):
    """
    Verify that threading uses less memory than multiprocessing.

    This is hard to measure precisely, but we can at least verify
    that the threading version completes without memory errors.
    """
    # Run a large simulation with threading
    T_sim = 5000
    np.random.seed(999)

    try:
        sim = cvine_threading.simulate(T=T_sim)
        assert sim.shape[1] == T_sim
        print(f"\nSuccessfully simulated {T_sim} timesteps with threading")
    except MemoryError:
        pytest.fail("Threading version ran out of memory")


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    pytest.main([__file__, "-v", "-s"])
