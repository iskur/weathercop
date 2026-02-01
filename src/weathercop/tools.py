import contextlib
import os
import shelve
import datetime
import time
from collections import UserDict
from hashlib import md5
from pathlib import Path
import dill
import numpy as np
import pandas as pd
import json

shelve.Pickler = dill.Pickler
shelve.Unpickler = dill.Unpickler


@contextlib.contextmanager
def acquire_file_lock(lock_file, timeout=60):
    """
    Acquire and release a file-based lock for cross-process synchronization.

    Uses atomic file creation to serialize access to shared resources across
    multiple processes (e.g., pytest-xdist workers). The lock is held for the
    duration of the context manager.

    Args:
        lock_file: Path object or string for the lock file to create/remove
        timeout: Maximum seconds to wait for lock acquisition (default: 60)

    Yields:
        None - lock is acquired when entering context, released on exit

    Raises:
        RuntimeError: If timeout exceeded while waiting for lock
    """
    lock_file = Path(lock_file)
    start_time = time.time()

    # Acquire lock via atomic file creation
    while True:
        try:
            with open(lock_file, 'x') as f:
                f.write(str(os.getpid()))
            break
        except FileExistsError:
            if time.time() - start_time > timeout:
                raise RuntimeError(
                    f"Timeout ({timeout}s) waiting for lock {lock_file}"
                )
            time.sleep(0.01)

    try:
        yield
    finally:
        # Release lock
        try:
            lock_file.unlink()
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def shelve_open(filename, *args, **kwds):
    """
    Context manager for shelve database access with file-based locking.

    Prevents concurrent access from multiple processes (e.g., pytest-xdist workers)
    from corrupting the SQLite WAL database. Uses atomic file creation for
    cross-process synchronization, similar to json_cache_open().

    Args:
        filename: Path to shelve database file
        *args, **kwds: Passed to shelve.open() (currently unused, reserved for future)

    Yields:
        shelve.Shelf: Open database object with exclusive access
    """
    filename = str(filename)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Create lock file path: store alongside the shelve database
    lock_file = Path(filename).parent / f".{Path(filename).name}.lock"

    # Acquire lock before opening database
    with acquire_file_lock(lock_file):
        sh = shelve.open(filename, "c")
        try:
            yield sh
        finally:
            sh.close()


@contextlib.contextmanager
def json_cache_open(filename):
    """
    Context manager for JSON-based expression caching with file-based locking.

    Loads JSON cache file, yields the cache dict, and saves back to disk.
    Only locks during actual file I/O (load/save), not during caller's usage,
    to minimize contention in high-parallelization scenarios (pytest-xdist -n 16+).

    Args:
        filename: Path to JSON cache file. Directory is created if missing.

    Yields:
        dict: The cache dictionary. Keys are cache IDs, values are serialized expressions.
    """
    filename = Path(filename)
    lock_file = filename.parent / f".{filename.name}.lock"

    # Create directory if needed
    if filename.parent != Path('.'):
        filename.parent.mkdir(parents=True, exist_ok=True)

    # Lock and load cache
    with acquire_file_lock(lock_file):
        cache = {}
        if filename.exists():
            try:
                with open(filename, 'r') as f:
                    cache = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError):
                # UnicodeDecodeError occurs when trying to read old binary shelve cache
                cache = {}

    # Yield WITHOUT lock - caller can modify freely (no contention)
    yield cache

    # Lock and save cache
    with acquire_file_lock(lock_file):
        with open(filename, 'w') as f:
            json.dump(cache, f, indent=2, sort_keys=True)


@contextlib.contextmanager
def chdir(dirname):
    """Temporarily change the working directory with a with-statement."""
    old_dir = os.path.abspath(os.path.curdir)
    if dirname:  # could be an empty string
        dirname = str(dirname)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        os.chdir(dirname)
    yield
    os.chdir(old_dir)


def gen_hash(string):
    return md5(string.encode()).hexdigest()


def hash_cop(cls):
    cop_expr = cls.cop_expr if hasattr(cls, "cop_expr") else cls
    return gen_hash(cop_expr.__repr__())


class ADict(UserDict):
    def __add__(self, other):
        # we need a copy to work with
        left_dict = dict(self)
        left_dict.update(other)
        # make sure we can do this operation also with the returned
        # object
        return ADict(left_dict)

    def __sub__(self, other):
        left_dict = dict(self)
        if type(other) is dict:
            del_keys = other.keys()
        elif type(other) is str:
            del_keys = (other,)
        else:
            del_keys = other
        for del_key in del_keys:
            del left_dict[del_key]
        return ADict(left_dict)
