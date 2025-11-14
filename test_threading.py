#!/usr/bin/env python
"""Test threading vs multiprocessing implementation."""
import numpy as np
import time
from weathercop import cop_conf

# Test with a simple simulation to verify threading works
print("=" * 60)
print("Testing Threading vs Multiprocessing Implementation")
print("=" * 60)

# Generate some test data
np.random.seed(42)
n_vars = 4
n_timesteps = 1000

# Create simple test ranks
ranks = np.random.uniform(0, 1, (n_vars, n_timesteps))

print(f"\nTest data shape: {ranks.shape}")
print(f"Number of workers: {cop_conf.n_nodes}")
print(f"Using threading: {cop_conf.USE_THREADING}")

# Test 1: Verify configuration
print("\n" + "="*60)
print("Configuration Test")
print("="*60)
print(f"PROFILE mode: {cop_conf.PROFILE}")
print(f"USE_THREADING: {cop_conf.USE_THREADING}")
print(f"n_nodes: {cop_conf.n_nodes}")

# Test 2: Basic imports
print("\n" + "="*60)
print("Import Test")
print("="*60)
try:
    from multiprocessing.pool import ThreadPool
    print("✓ ThreadPool imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ThreadPool: {e}")

try:
    import multiprocessing
    print(f"✓ multiprocessing imported successfully (cpu_count={multiprocessing.cpu_count()})")
except ImportError as e:
    print(f"✗ Failed to import multiprocessing: {e}")

# Test 3: Simple parallel test
print("\n" + "="*60)
print("Parallel Execution Test")
print("="*60)

def simple_task(x):
    """Simple CPU-bound task."""
    return np.sum(np.sin(np.linspace(0, x, 1000)))

test_data = list(range(100))

# Test with threading
print("\nTesting with ThreadPool:")
start = time.time()
with ThreadPool(cop_conf.n_nodes) as pool:
    results_thread = pool.map(simple_task, test_data)
time_thread = time.time() - start
print(f"  Time: {time_thread:.3f}s")
print(f"  Results mean: {np.mean(results_thread):.3f}")

# Test with multiprocessing
print("\nTesting with multiprocessing.Pool:")
start = time.time()
with multiprocessing.Pool(cop_conf.n_nodes) as pool:
    results_mp = pool.map(simple_task, test_data)
time_mp = time.time() - start
print(f"  Time: {time_mp:.3f}s")
print(f"  Results mean: {np.mean(results_mp):.3f}")

# Verify results are identical
if np.allclose(results_thread, results_mp):
    print("\n✓ Results are identical between threading and multiprocessing")
else:
    print("\n✗ Results differ between threading and multiprocessing!")
    print(f"  Max difference: {np.max(np.abs(np.array(results_thread) - np.array(results_mp)))}")

print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Threading speedup vs multiprocessing: {time_mp/time_thread:.2f}x")
if time_thread < time_mp:
    print("✓ Threading is faster (as expected for numpy operations)")
else:
    print("⚠ Multiprocessing is faster (may indicate GIL contention)")

print("\n" + "="*60)
print("Test Complete")
print("="*60)
