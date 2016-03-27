import numpy as np
from scipy import stats as spstats


def rel_ranks(data, method="average"):
    return (spstats.rankdata(data, method) - .5) / len(data)


def spearmans_rank(rel_ranks_x, rel_ranks_y):
    """Spearmans rho of relative ranks."""
    n = len(rel_ranks_x)
    return (12. / (n * (n + 1) * (n - 1)) *
            np.sum((rel_ranks_x * n + .5) *
                   (rel_ranks_y * n + .5)) -
            3. * (n + 1) / (n - 1))


def asymmetry1(uu, vv):
    """
    >>> import numpy as np
    >>> uu, vv = np.ones(100), np.ones(100)
    >>> asymmetry1(uu, vv)
    0.25
    """
    xx = uu - .5
    yy = vv - .5
    return np.mean(xx * yy * (xx + yy))


def asymmetry2(uu, vv):
    """
    >>> import numpy as np
    >>> uu, vv = np.zeros(100), np.ones(100)
    >>> asymmetry2(uu, vv)
    -0.25
    """
    xx = uu - .5
    yy = vv - .5
    return np.mean(-xx * yy * (xx - yy))
