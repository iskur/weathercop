#!/usr/bin/env python
"""
Benchmark CVine performance: Threading vs Multiprocessing

This script measures the actual speedup from using threading instead of
multiprocessing for CVine simulation, and verifies that means are preserved.
"""
import time
import numpy as np
import scipy.stats as spstats
from weathercop import cop_conf
from weathercop.vine import CVine


def create_test_data(n_vars=5, n_timesteps=2000, seed=42):
    """Generate realistic correlated data for testing."""
    np.random.seed(seed)

    # Create realistic correlation structure
    corr_matrix = np.eye(n_vars)
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            # Decreasing correlation with distance
            corr = 0.7 * np.exp(-0.3 * abs(i - j))
            corr_matrix[i, j] = corr_matrix[j, i] = corr

    # Generate correlated data
    mean = np.zeros(n_vars)
    data = np.random.multivariate_normal(mean, corr_matrix, n_timesteps).T

    # Convert to ranks (uniform margins)
    ranks = np.array([spstats.rankdata(row)/(len(row)+1) for row in data])

    return ranks


def benchmark_simulation(vine, T_sim, n_trials=3, label=""):
    """Benchmark vine simulation performance."""
    times = []

    for trial in range(n_trials):
        np.random.seed(123 + trial)  # Different seed each trial
        start = time.time()
        sim = vine.simulate(T=T_sim)
        elapsed = time.time() - start
        times.append(elapsed)

        # Verify simulation completed
        assert sim.shape[1] == T_sim, f"Simulation shape mismatch"

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n{label}:")
    print(f"  Average time: {avg_time:.3f}s ± {std_time:.3f}s ({n_trials} trials)")
    print(f"  Throughput:   {T_sim/avg_time:.0f} timesteps/sec")

    return avg_time, times


def verify_means_preserved(ranks_obs, ranks_sim, varnames, tolerance=0.05):
    """
    Verify that simulated means match observed means.

    This is the CRITICAL requirement from the user.
    """
    obs_means = ranks_obs.mean(axis=1)
    sim_means = ranks_sim.mean(axis=1)

    print("\n" + "="*70)
    print("CRITICAL TEST: Simulated Means vs Observed Means")
    print("="*70)
    print(f"{'Variable':<12} {'Observed':<12} {'Simulated':<12} {'Diff %':<10} {'Status'}")
    print("-"*70)

    all_passed = True
    for i, varname in enumerate(varnames):
        obs = obs_means[i]
        sim = sim_means[i]
        diff_pct = (sim - obs) / obs * 100

        status = "✓ PASS" if abs(diff_pct) < tolerance * 100 else "✗ FAIL"
        if abs(diff_pct) >= tolerance * 100:
            all_passed = False

        print(f"{varname:<12} {obs:<12.6f} {sim:<12.6f} {diff_pct:<10.2f} {status}")

    print("-"*70)
    if all_passed:
        print("✓ All means match within tolerance!")
    else:
        print("✗ Some means exceed tolerance!")

    return all_passed


def main():
    print("="*70)
    print("CVine Performance Benchmark: Threading vs Multiprocessing")
    print("="*70)

    # Configuration
    n_vars = 5
    n_obs_timesteps = 2000
    n_sim_timesteps = 1000
    n_trials = 3

    # Create test data
    print(f"\nGenerating test data...")
    print(f"  Variables: {n_vars}")
    print(f"  Observation timesteps: {n_obs_timesteps}")
    print(f"  Simulation timesteps: {n_sim_timesteps}")

    ranks = create_test_data(n_vars, n_obs_timesteps)
    varnames = [f'var{i+1}' for i in range(n_vars)]

    # Build vine (same for both tests)
    print(f"\nBuilding CVine...")
    print(f"  Using Kendall's tau weights (faster)")
    print(f"  Central node: {varnames[0]}")

    build_start = time.time()
    vine_test = CVine(
        ranks,
        varnames=varnames,
        verbose=False,
        weights='tau',  # Faster than likelihood
        central_node=varnames[0]
    )
    build_time = time.time() - build_start
    print(f"  Build time: {build_time:.2f}s")

    # =======================================================================
    # BENCHMARK 1: Multiprocessing (baseline)
    # =======================================================================
    print("\n" + "="*70)
    print("BENCHMARK 1: Multiprocessing (Baseline)")
    print("="*70)

    cop_conf.USE_THREADING = False
    vine_mp = CVine(ranks, varnames=varnames, verbose=False, weights='tau',
                    central_node=varnames[0])

    time_mp, times_mp = benchmark_simulation(
        vine_mp, n_sim_timesteps, n_trials, "Multiprocessing"
    )

    # Get simulation for mean verification
    np.random.seed(999)
    sim_mp = vine_mp.simulate(T=n_sim_timesteps)

    # =======================================================================
    # BENCHMARK 2: Threading
    # =======================================================================
    print("\n" + "="*70)
    print("BENCHMARK 2: Threading")
    print("="*70)

    cop_conf.USE_THREADING = True
    vine_thread = CVine(ranks, varnames=varnames, verbose=False, weights='tau',
                        central_node=varnames[0])

    time_thread, times_thread = benchmark_simulation(
        vine_thread, n_sim_timesteps, n_trials, "Threading"
    )

    # Get simulation for mean verification
    np.random.seed(999)
    sim_thread = vine_thread.simulate(T=n_sim_timesteps)

    # =======================================================================
    # RESULTS
    # =======================================================================
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)

    speedup = time_mp / time_thread
    print(f"\nMultiprocessing: {time_mp:.3f}s")
    print(f"Threading:       {time_thread:.3f}s")
    print(f"\nSpeedup:         {speedup:.2f}x")

    if speedup > 1.0:
        improvement = (speedup - 1) * 100
        print(f"Performance gain: {improvement:.1f}%")
        print(f"\n✓ Threading is {improvement:.1f}% faster!")
    elif speedup > 0.9:
        print(f"\n≈ Threading is approximately the same speed")
    else:
        slowdown = (1 - speedup) * 100
        print(f"Performance loss: {slowdown:.1f}%")
        print(f"\n✗ Threading is {slowdown:.1f}% slower")

    # =======================================================================
    # VERIFY MEANS PRESERVED
    # =======================================================================
    means_ok = verify_means_preserved(ranks, sim_thread, varnames, tolerance=0.05)

    # =======================================================================
    # VERIFY THREADING & MULTIPROCESSING GIVE SAME RESULTS
    # =======================================================================
    print("\n" + "="*70)
    print("CORRECTNESS: Threading vs Multiprocessing Results")
    print("="*70)

    max_diff = np.max(np.abs(sim_thread - sim_mp))
    print(f"\nMaximum absolute difference: {max_diff:.2e}")

    if max_diff < 1e-4:
        print("✓ Results are identical (within numerical precision)")
    else:
        print("✗ Results differ - this indicates a problem!")

    # =======================================================================
    # SUMMARY
    # =======================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n1. Performance:  Threading is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    print(f"2. Correctness:  {'✓ PASS' if max_diff < 1e-4 else '✗ FAIL'}")
    print(f"3. Means match:  {'✓ PASS' if means_ok else '✗ FAIL'}")

    if speedup > 1.0 and max_diff < 1e-4 and means_ok:
        print("\n✓✓✓ Threading is recommended! ✓✓✓")
    elif speedup > 0.9 and max_diff < 1e-4 and means_ok:
        print("\n✓ Threading works correctly (similar performance)")
    else:
        print("\n⚠ Further investigation needed")

    # Restore default
    cop_conf.USE_THREADING = True

    return speedup, means_ok


if __name__ == "__main__":
    speedup, means_ok = main()
    print("\nBenchmark complete!")
