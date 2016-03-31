import contextlib
import os
import shelve
from collections import UserDict
from hashlib import md5


@contextlib.contextmanager
def shelve_open(filename, *args, **kwds):
    sh = shelve.open(filename, "c")
    yield sh
    sh.close()


@contextlib.contextmanager
def chdir(dirname):
    """Temporarily change the working directory with a with-statement."""
    old_dir = os.path.abspath(os.path.curdir)
    if dirname:  # could be an empty string
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        os.chdir(dirname)
    yield
    os.chdir(old_dir)


def hash_cop(cls):
    cop_expr = cls.cop_expr if hasattr(cls, "cop_expr") else cls
    return md5(cop_expr.__repr__().encode()).hexdigest()


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
            del_keys = other,
        else:
            del_keys = other
        for del_key in del_keys:
            del left_dict[del_key]
        return ADict(left_dict)


