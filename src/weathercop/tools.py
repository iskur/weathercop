import contextlib
import os
import shelve
import datetime
from collections import UserDict
from hashlib import md5
import dill
import numpy as np
import pandas as pd
import json

shelve.Pickler = dill.Pickler
shelve.Unpickler = dill.Unpickler


@contextlib.contextmanager
def shelve_open(filename, *args, **kwds):
    filename = str(filename)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    sh = shelve.open(filename, "c")
    yield sh
    sh.close()


@contextlib.contextmanager
def json_cache_open(filename):
    """
    Context manager for JSON-based expression caching.

    Loads a JSON cache file on entry, yields the cache dict for read/write,
    and saves back to disk on exit.

    Args:
        filename: Path to JSON cache file. Directory is created if missing.

    Yields:
        dict: The cache dictionary. Keys are cache IDs, values are serialized expressions.
    """
    filename = str(filename)
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    # Load existing cache or start empty
    cache = {}
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                cache = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError):
            # UnicodeDecodeError occurs when trying to read old binary shelve cache
            cache = {}

    yield cache

    # Save back to file with indentation for readability
    with open(filename, 'w') as f:
        json.dump(cache, f, indent=2)


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
