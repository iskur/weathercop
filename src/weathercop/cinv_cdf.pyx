# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
# from ctypes import CFUNCTYPE
# from libc.stdlib cimport malloc, free
from cython.parallel import prange, parallel

cimport numpy as np
cimport cython

ctypedef np.float_t DTYPE_t
# ctypedef double (*CONDITIONAL_FUNC)(double[::1] rank_u_ar,
#                                     double[::1] rank_v_ar,
#                                     object theta)
# ctypedef double (*CONDITIONAL_FUNC_PRIME)(double[::1] rank_u_ar,
#                                           double[::1] rank_v_ar,
#                                           object theta)
# ctypedef double (*CFUNCTYPE)(double[::1] rank_u_ar,
#                              double[::1] rank_v_ar,
#                              object theta)
# ctypedef double (*CFUNCTYPE)(double[::1] rank_u_ar,
#                              double[::1] rank_v_ar,
#                              object theta)
# CWRAPPER = CFUNCTYPE(None)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef double[::1] _newton(CONDITIONAL_FUNC conditional_func,
#                          CONDITIONAL_FUNC_PRIME conditional_func_prime,
#                          np.ndarray[DTYPE_t, ndim=1] ranks1,
#                          np.ndarray[DTYPE_t, ndim=1] quantiles,
#                          np.ndarray[DTYPE_t, ndim=2] thetas,
#                          int given_v):
#     cdef int i, it
#     cdef int max_it = 100
#     cdef int n_steps = len(ranks1)
#     cdef double zk, zkp1, zkp1_prelim, rank0, rank1, quantile, theta
#     cdef double gz, gz_prime, step
#     cdef double eps = 1e-4
#     cdef double[::1] ranks2 = np.empty_like(ranks1)
#     cdef double[:, ::1] theta_ar = np.empty((thetas.shape[0], 1))
#     cdef double[::1] rank_u_ar = np.empty((1,))
#     cdef double[::1] rank_v_ar = np.empty((1,))
#     # its = np.zeros_like(ranks2)
#     for i in range(n_steps):
#         rank1 = ranks1[i]
#         quantile = quantiles[i]
#         # theta = thetas[:, i]
#         # theta_ar[:, 0] = theta
#         theta_ar[:, 0] = thetas[:, i]
#         rank0 = max(eps, min(quantile, 1 - eps))
#         zk = 99.
#         zkp1 = rank0
#         it = 0
#         while (abs(zk - zkp1) > 1e-6):
#             zk = zkp1
#             if given_v == 1:
#                 rank_u_ar[0] = zk
#                 rank_v_ar[0] = rank1
#             else:
#                 rank_u_ar[0] = rank1
#                 rank_v_ar[0] = zk
#             gz = conditional_func(rank_u_ar,
#                                   rank_v_ar,
#                                   theta_ar)
#             gz_prime = conditional_func_prime(rank_u_ar,
#                                               rank_v_ar,
#                                               theta_ar)
#             if gz_prime == 0:
#                 step = 0  # this will end the loop
#             else:
#                 step = (gz - quantile) / gz_prime
#             zkp1_prelim = zk - step
#             if zkp1_prelim > 1:
#                 step = -.5 * (max_it - it) / max_it * (1 - zk)
#             elif zkp1_prelim < 0:
#                 step = .5 * (max_it - it) / max_it * zk
#             zkp1 = zk - step
#             it += 1
#             if it == max_it:
#                 break
#         ranks2[i] = zkp1
#     return ranks2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[::1] _newton(conditional_func,
                         conditional_func_prime,
                         np.ndarray[DTYPE_t, ndim=1] ranks1,
                         np.ndarray[DTYPE_t, ndim=1] quantiles,
                         np.ndarray[DTYPE_t, ndim=2] thetas,
                         int given_v):
    cdef unsigned int i, it
    cdef unsigned int max_it = 100
    cdef unsigned int n_steps = len(ranks1)
    cdef double zk, zkp1, zkp1_prelim, rank0, rank1, quantile, theta
    cdef double gz, gz_prime, step
    cdef double eps = 1e-4
    cdef double[::1] ranks2 = np.empty_like(ranks1)
    cdef np.ndarray[DTYPE_t, ndim=2] theta_ar = np.empty((thetas.shape[0],
                                                          1))
    cdef np.ndarray[DTYPE_t, ndim=1] rank_u_ar = np.empty((1,))
    cdef np.ndarray[DTYPE_t, ndim=1] rank_v_ar = np.empty((1,))
    for i in range(n_steps):
        rank1 = ranks1[i]
        quantile = quantiles[i]
        theta_ar[:, 0] = thetas[:, i]
        rank0 = max(eps, min(quantile, 1 - eps))
        zk = 99.
        zkp1 = rank0
        it = 0
        while (abs(zk - zkp1) > 1e-6):
            zk = zkp1
            if given_v == 1:
                rank_u_ar[0] = zk
                rank_v_ar[0] = rank1
            else:
                rank_u_ar[0] = rank1
                rank_v_ar[0] = zk
            # Extract the current row of theta as a 1D array (avoids deprecation warnings)
            theta_1d = np.ascontiguousarray(theta_ar[:, 0])
            gz = conditional_func(rank_u_ar,
                                  rank_v_ar,
                                  theta_1d)
            gz_prime = conditional_func_prime(rank_u_ar,
                                              rank_v_ar,
                                              theta_1d)
            if gz_prime == 0:
                step = 0  # this will end the loop
            else:
                step = (gz - quantile) / gz_prime
            zkp1_prelim = zk - step
            if zkp1_prelim > 1:
                step = -.5 * (max_it - it) / max_it * (1 - zk)
            elif zkp1_prelim < 0:
                step = .5 * (max_it - it) / max_it * zk
            zkp1 = zk - step
            it += 1
            if it == max_it:
                break
        ranks2[i] = zkp1
    return ranks2

cpdef newton(conditional_func, conditional_func_prime, ranks1, quantiles,
            thetas, given_v):
    ranks2 = _newton(conditional_func,
                     conditional_func_prime,
                     ranks1,
                     quantiles, thetas, int(given_v))
    return np.array(ranks2)

