# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
from libc.math cimport sqrt, erf as erf_scalar
cimport numpy as np

from scipy.special import erfinv as erfinv_scalar

cpdef norm_inv_cdf_given_u_(
        np.ndarray[double, ndim=1] uu,
        np.ndarray[double, ndim=1] qq,
        np.ndarray[double, ndim=1] rho,
        np.ndarray[double, ndim=1] result):

    cdef int n = len(uu)
    cdef int i

    cdef double q_inv, u_inv, sqrt_term, tmp

    for i in range(n):
        u_inv = sqrt(2) * erfinv_scalar(2 * uu[i] - 1)
        q_inv = sqrt(2) * erfinv_scalar(2 * qq[i] - 1)
        sqrt_term = sqrt(1 - rho[i] ** 2)
        tmp = q_inv * sqrt_term + rho[i] * u_inv
        result[i] = 0.5 * (1 + erf_scalar(tmp / sqrt(2)))


cpdef norm_cdf_given_u(
        np.ndarray[double, ndim=1] uu,
        np.ndarray[double, ndim=1] vv,
        np.ndarray[double, ndim=1] rho,
        np.ndarray[double, ndim=1] result):

    cdef int n = len(uu)
    cdef int i

    cdef double e_u, e_v, sqrt_term, tmp

    for i in range(n):
        e_u = erfinv_scalar(2 * uu[i] - 1)
        e_v = erfinv_scalar(2 * vv[i] - 1)
        sqrt_term = sqrt(1 - rho[i] ** 2)
        tmp = (rho[i] * e_u - e_v) / sqrt_term
        result[i] = 0.5 * (1 + erf_scalar(tmp))


cpdef norm_inv_cdf_given_u(
        np.ndarray[double, ndim=1] uu,
        np.ndarray[double, ndim=1] qq,
        np.ndarray[double, ndim=1] rho,
        np.ndarray[double, ndim=1] result):

    cdef int n = len(uu)
    cdef int i

    cdef double e_u, e_v, sqrt_term, tmp

    for i in range(n):
        e_u = erfinv_scalar(2 * uu[i] - 1)
        e_v = erfinv_scalar(1 - 2 * qq[i])
        sqrt_term = sqrt(1 - rho[i] ** 2)
        tmp = rho[i] * e_u - e_v * sqrt_term
        result[i] = 0.5 * (1 + erf_scalar(tmp))


cpdef erf(np.ndarray[double, ndim=1] x,
          np.ndarray[double, ndim=1] result):
    cdef int n = len(x)
    cdef int i
    for i in range(n):
        result[i] = erf_scalar(x[i])


cpdef erfinv(np.ndarray[double, ndim=1] x,
             np.ndarray[double, ndim=1] result):
    cdef int n = len(x)
    cdef int i
    for i in range(n):
        result[i] = erfinv_scalar(x[i])
