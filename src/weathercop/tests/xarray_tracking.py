"""Track xarray dataset lifecycle to detect leaks."""
import xarray as xr
import weakref


class XarrayTracker:
    """Track active xarray datasets and their lifecycle.

    Uses id-based tracking with weakref finalization callbacks instead of WeakSet
    to handle unhashable Dataset objects in Python 3.13+.
    """

    def __init__(self):
        """Initialize tracker with empty dataset registry."""
        # Track datasets by id() since xarray Datasets are not hashable
        self.dataset_ids = set()
        self.dataset_refs = {}  # id -> weakref mapping
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
            tracker._track_dataset(ds)
            return ds

        def tracked_open_mf(*args, **kwargs):
            ds = original_open_mf(*args, **kwargs)
            tracker._track_dataset(ds)
            return ds

        xr.open_dataset = tracked_open
        xr.open_mfdataset = tracked_open_mf
        self._hook_installed = True

    def _track_dataset(self, ds):
        """Track a dataset using weak references and id-based tracking."""
        ds_id = id(ds)
        self.dataset_ids.add(ds_id)

        # Create a finalization callback to remove the dataset when it's garbage collected
        def remove_on_gc():
            self.dataset_ids.discard(ds_id)
            self.dataset_refs.pop(ds_id, None)

        # Store weak reference with finalization callback
        try:
            ref = weakref.ref(ds, lambda r: remove_on_gc())
            self.dataset_refs[ds_id] = ref
        except TypeError:
            # If weakref fails, just track the id and rely on periodic cleanup
            pass

    def count_active(self):
        """Get count of active xarray datasets."""
        # Clean up dead references
        self.dataset_ids = {
            ds_id for ds_id in self.dataset_ids
            if ds_id in self.dataset_refs and self.dataset_refs[ds_id]() is not None
        }
        return len(self.dataset_ids)

    def report(self):
        """Get detailed report of active datasets."""
        # Clean up dead references
        active_datasets = []
        dead_ids = set()

        for ds_id, ref in list(self.dataset_refs.items()):
            ds = ref()
            if ds is None:
                dead_ids.add(ds_id)
            else:
                active_datasets.append(ds)

        # Remove dead references
        for ds_id in dead_ids:
            self.dataset_ids.discard(ds_id)
            self.dataset_refs.pop(ds_id, None)

        report = {
            'count': len(active_datasets),
            'datasets': [
                {
                    'dims': list(ds.dims),
                    'data_vars': list(ds.data_vars),
                    'memory_mb': sum(
                        var.nbytes / 1024 / 1024
                        for var in ds.data_vars.values()
                    ) if hasattr(ds, 'data_vars') else 0,
                }
                for ds in active_datasets
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
