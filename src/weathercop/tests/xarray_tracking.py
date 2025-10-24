"""Track xarray dataset lifecycle to detect leaks."""
import xarray as xr
import weakref
import gc
from pathlib import Path


class XarrayTracker:
    """Track active xarray datasets and their lifecycle."""

    def __init__(self):
        """Initialize tracker with empty dataset registry."""
        self.datasets = weakref.WeakSet()
        self._hook_installed = False

    def install_hook(self):
        """Install hook to track xarray dataset creation."""
        if self._hook_installed:
            return

        # Monkey-patch xr.open_dataset and xr.open_mfdataset
        original_open = xr.open_dataset
        original_open_mf = xr.open_mfdataset

        tracker = self

        def tracked_open(*args, **kwargs):
            ds = original_open(*args, **kwargs)
            tracker.datasets.add(ds)
            return ds

        def tracked_open_mf(*args, **kwargs):
            ds = original_open_mf(*args, **kwargs)
            tracker.datasets.add(ds)
            return ds

        xr.open_dataset = tracked_open
        xr.open_mfdataset = tracked_open_mf
        self._hook_installed = True

    def count_active(self):
        """Get count of active xarray datasets."""
        return len(self.datasets)

    def report(self):
        """Get detailed report of active datasets."""
        active = list(self.datasets)
        report = {
            'count': len(active),
            'datasets': [
                {
                    'dims': list(ds.dims),
                    'data_vars': list(ds.data_vars),
                    'memory_mb': sum(
                        var.nbytes / 1024 / 1024
                        for var in ds.data_vars.values()
                    ) if hasattr(ds, 'data_vars') else 0,
                }
                for ds in active
            ]
        }
        return report


# Global tracker instance
_xarray_tracker = None


def get_xarray_tracker():
    """Get or create global xarray tracker."""
    global _xarray_tracker
    if _xarray_tracker is None:
        _xarray_tracker = XarrayTracker()
        _xarray_tracker.install_hook()
    return _xarray_tracker
