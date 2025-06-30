import contextlib
import os
import shelve
import datetime
from collections import UserDict
from hashlib import md5
import dill
import numpy as np
import pandas as pd

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
