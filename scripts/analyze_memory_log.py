#!/usr/bin/env python
"""Analyze memory diagnostics log after OOM incident.

Usage:
    python scripts/analyze_memory_log.py ~/.weathercop/test_memory.log
"""
import sys
from pathlib import Path


def parse_log_file(log_path):
    """Parse memory diagnostics log file.

    Returns:
        List of log entries (dicts with timestamp, event, test, memory info)
    """
    entries = []
    current_entry = None

    with open(log_path) as f:
        for line in f:
            line = line.rstrip()

            if not line:
                continue

            # New entry starts with timestamp
            if line.startswith("["):
                if current_entry:
                    entries.append(current_entry)

                # Parse: [timestamp] EVENT: test_name
                parts = line.split("]", 1)[1].strip().split(":", 1)
                event = parts[0].strip()
                test = parts[1].strip() if len(parts) > 1 else ""

                current_entry = {
                    'raw_line': line,
                    'event': event,
                    'test': test,
                    'details': []
                }
            else:
                # Continuation line - add to current entry
                if current_entry:
                    current_entry['details'].append(line.strip())

        if current_entry:
            entries.append(current_entry)

    return entries


def analyze_memory_growth(entries):
    """Analyze memory growth across tests."""
    memory_by_test = {}

    for entry in entries:
        if entry['event'] != "TEST_END":
            continue

        test_name = entry['test']
        peak_mb = None

        # Extract peak memory from details
        for detail in entry['details']:
            if "Peak memory:" in detail:
                # Parse: Peak memory: 1234.5 MB
                try:
                    peak_mb = float(detail.split(":")[-1].strip().split()[0])
                except ValueError:
                    pass

        if peak_mb:
            memory_by_test[test_name] = peak_mb

    # Sort by memory usage
    sorted_tests = sorted(memory_by_test.items(), key=lambda x: x[1], reverse=True)

    print("\n=== Top 10 Memory-Intensive Tests ===")
    for i, (test, peak_mb) in enumerate(sorted_tests[:10], 1):
        print(f"{i:2d}. {peak_mb:8.1f} MB: {test}")

    if sorted_tests:
        total_mb = sum(m for _, m in sorted_tests)
        avg_mb = total_mb / len(sorted_tests)
        max_mb = sorted_tests[0][1]
        print(f"\nAverage: {avg_mb:.1f} MB")
        print(f"Max: {max_mb:.1f} MB")
        print(f"Total tests with peak memory: {len(sorted_tests)}")


def find_oom_point(entries):
    """Find where memory exhaustion likely occurred."""
    print("\n=== Memory Growth Timeline ===")

    prev_peak = 0
    spike_tests = []

    for entry in entries:
        if entry['event'] != "TEST_END":
            continue

        test_name = entry['test']
        peak_mb = None

        for detail in entry['details']:
            if "Peak memory:" in detail:
                try:
                    peak_mb = float(detail.split(":")[-1].strip().split()[0])
                except ValueError:
                    pass

        if peak_mb:
            growth = peak_mb - prev_peak
            if growth > 100:  # Spike of >100MB
                spike_tests.append((test_name, peak_mb, growth))
            prev_peak = peak_mb

    if spike_tests:
        print("\nLarge memory spikes (>100MB):")
        for test, peak, growth in spike_tests:
            print(f"  {test}: {peak:.1f} MB (↑ {growth:.1f} MB)")

        print(f"\nLikely culprit: {spike_tests[-1][0]}")
    else:
        print("No major spikes detected. Memory grew gradually.")


def analyze_xarray_leaks(entries):
    """Analyze xarray dataset leaks over time."""
    print("\n=== XArray Dataset Tracking ===")

    xarray_counts = []
    for entry in entries:
        if entry['event'] != "TEST_END":
            continue

        test_name = entry['test']
        xarray_count = 0

        for detail in entry['details']:
            if "Open xarray datasets:" in detail:
                try:
                    xarray_count = int(detail.split(":")[-1].strip())
                except ValueError:
                    pass

        if xarray_count > 0:
            xarray_counts.append((test_name, xarray_count))

    if xarray_counts:
        print(f"\nTests with unclosed xarray datasets:")
        for test, count in xarray_counts:
            print(f"  {test}: {count} open dataset(s)")

        if xarray_counts[-1][1] > xarray_counts[0][1]:
            print("\n⚠️ XArray count is INCREASING - likely a leak!")
    else:
        print("\n✓ No unclosed xarray datasets detected")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_memory_log.py <log_file>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Error: {log_path} not found")
        sys.exit(1)

    print(f"Analyzing: {log_path}")
    entries = parse_log_file(log_path)
    print(f"Total log entries: {len(entries)}")

    analyze_memory_growth(entries)
    find_oom_point(entries)
    analyze_xarray_leaks(entries)


if __name__ == "__main__":
    main()
