"""Memory diagnostics and logging for test execution.

This module provides unbuffered, real-time memory tracking during tests.
Logs are written immediately to disk (not buffered) so they survive OOM kills.
"""
import os
import gc
import psutil
import tracemalloc
from pathlib import Path
from datetime import datetime


class MemoryDiagnosticsLogger:
    """Unbuffered logger for memory diagnostics during test execution."""

    def __init__(self, log_path=None):
        """Initialize the logger.

        Args:
            log_path: Path to write logs. Defaults to WEATHERCOP_DIR/test_memory.log
        """
        if log_path is None:
            weathercop_dir = Path(os.environ.get(
                'WEATHERCOP_DIR',
                Path.home() / '.weathercop'
            ))
            log_path = weathercop_dir / "test_memory.log"

        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(exist_ok=True, parents=True)

        # Clear previous log at session start
        self.log_path.write_text("")

        self.process = psutil.Process(os.getpid())
        self.session_start = datetime.now()

    def log_event(self, event_type, test_name, details=None):
        """Write an unbuffered log entry immediately to disk.

        Args:
            event_type: str - "SESSION_START", "TEST_START", "TEST_END", "SESSION_END"
            test_name: str - Name of the test
            details: dict - Additional data to log
        """
        timestamp = datetime.now()
        elapsed = (timestamp - self.session_start).total_seconds()

        # Get process memory from /proc
        try:
            with open("/proc/self/status") as f:
                proc_status = {}
                for line in f:
                    if line.startswith(("VmPeak", "VmHWM", "VmRSS", "VmSize")):
                        key, val = line.rstrip().split(":\t")
                        proc_status[key] = val.strip()
        except FileNotFoundError:
            proc_status = {"error": "proc/status not available"}

        # Get Python tracemalloc snapshot (only if actively tracking)
        try:
            if tracemalloc.is_tracing():
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')[:3]
                tracemalloc_info = [
                    f"{stat.filename}:{stat.lineno}: {stat.size / 1024 / 1024:.1f} MB"
                    for stat in top_stats
                ]
            else:
                tracemalloc_info = ["(tracemalloc not active)"]
        except Exception as e:
            tracemalloc_info = [f"error: {e}"]

        # Build log entry
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "elapsed_sec": elapsed,
            "event": event_type,
            "test": test_name,
            "proc_status": proc_status,
            "tracemalloc_top3": tracemalloc_info,
        }

        if details:
            log_entry.update(details)

        # Write immediately (unbuffered) with newline and flush
        log_line = self._format_entry(log_entry)
        with open(self.log_path, "a", buffering=1) as f:  # Line buffering
            f.write(log_line + "\n")
            f.flush()
            os.fsync(f.fileno())  # Force disk sync

    def _format_entry(self, entry):
        """Format log entry as readable text."""
        lines = [
            f"[{entry['timestamp']}] {entry['event']}: {entry['test']}",
            f"  Elapsed: {entry['elapsed_sec']:.1f}s",
        ]

        if entry.get('proc_status'):
            lines.append("  /proc/self/status:")
            for key, val in entry['proc_status'].items():
                lines.append(f"    {key}: {val}")

        if entry.get('tracemalloc_top3'):
            lines.append("  Top tracemalloc allocations:")
            for info in entry['tracemalloc_top3']:
                lines.append(f"    {info}")

        if entry.get('peak_memory_mb') is not None:
            lines.append(f"  Peak memory: {entry['peak_memory_mb']:.1f} MB")

        if entry.get('gc_stats'):
            lines.append(f"  GC: collections={entry['gc_stats']['collections']}, "
                        f"uncollectable={entry['gc_stats']['uncollectable']}")

        if entry.get('xarray_count'):
            lines.append(f"  Open xarray datasets: {entry['xarray_count']}")

        return "\n".join(lines)

    def log_session_start(self):
        """Log session start."""
        self.log_event("SESSION_START", "pytest", {
            "python_version": __import__('sys').version,
            "hostname": __import__('socket').gethostname(),
        })

    def log_test_start(self, test_name):
        """Log test start and current memory state."""
        gc_stats = {
            'collections': sum(gc.get_count()),
            'uncollectable': len(gc.garbage),
        }
        self.log_event("TEST_START", test_name, {"gc_stats": gc_stats})

    def log_test_end(self, test_name, peak_memory_mb=None, xarray_count=0):
        """Log test end with memory snapshot."""
        gc_stats = {
            'collections': sum(gc.get_count()),
            'uncollectable': len(gc.garbage),
        }
        details = {
            "gc_stats": gc_stats,
            "xarray_count": xarray_count,
        }
        if peak_memory_mb is not None:
            details["peak_memory_mb"] = peak_memory_mb

        self.log_event("TEST_END", test_name, details)

    def log_session_end(self):
        """Log session end."""
        self.log_event("SESSION_END", "pytest")

    def start_peak_tracking(self):
        """Start tracking peak memory usage for current test."""
        tracemalloc.start()

    def get_peak_and_stop_tracking(self):
        """Get peak memory usage and stop tracking.

        Returns:
            Peak memory in MB, or None if tracemalloc not running.
        """
        try:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak / 1024 / 1024  # Convert to MB
        except ValueError:  # tracemalloc not started
            return None


# Global logger instance
_memory_logger = None


def get_memory_logger():
    """Get or create the global memory logger.

    Can be disabled by setting WEATHERCOP_DISABLE_DIAGNOSTICS=1 environment variable.
    """
    global _memory_logger
    if os.environ.get("WEATHERCOP_DISABLE_DIAGNOSTICS") == "1":
        return _NoOpLogger()  # Return no-op implementation
    if _memory_logger is None:
        _memory_logger = MemoryDiagnosticsLogger()
    return _memory_logger


class _NoOpLogger:
    """No-op logger that does nothing (used when diagnostics are disabled)."""
    def log_event(self, *args, **kwargs): pass
    def log_session_start(self): pass
    def log_test_start(self, *args): pass
    def log_test_end(self, *args, **kwargs): pass
    def log_session_end(self): pass
    def start_peak_tracking(self): pass
    def get_peak_and_stop_tracking(self): return None
