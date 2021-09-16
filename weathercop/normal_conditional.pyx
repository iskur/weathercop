# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import prange, parallel

cimport numpy as np


cdef extern from "mkl.h" nogil:
    void vdErf(int n,
               double *a,
               double *y)

    void vdErfInv(int n,
                  double *a,
                  double *y)

    void vdSqrt(int n,
                double *a,
                double *y)

    void vdCdfNorm(int n,
                   double *a,
                   double *y)

    void vdCdfNormInv(int n,
                      double *a,
                      double *y)


cpdef norm_inv_cdf_given_u_(double[:] uu,
                            double[:] qq,
                            double[:] rho,
                            double[:] result):
    cdef int n = len(uu)
    cdef int i
    q_inv = <double *> malloc(sizeof(double) * n)
    u_inv = <double *> malloc(sizeof(double) * n)
    sqrt = <double *> malloc(sizeof(double) * n)
    tmp_in = <double *> malloc(sizeof(double) * n)
    
    vdCdfNormInv(<int> n, &qq[0], &q_inv[0])
    vdCdfNormInv(<int> n, &uu[0], &u_inv[0])
    
    for i in xrange(n):
        tmp_in[i] = 1 - rho[i] ** 2
    vdSqrt(<int> n, &tmp_in[0], &sqrt[0])

    for i in xrange(n):
        tmp_in[i] = (q_inv[i] * sqrt[i] + rho[i] * u_inv[i])
    vdCdfNorm(<int> n, &tmp_in[0], &result[0])

    free(q_inv)
    free(u_inv)
    free(sqrt)
    free(tmp_in)


cpdef norm_cdf_given_u(double[:] uu,
                       double[:] vv,
                       double[:] rho,
                       double[:] result):
    cdef int n = len(uu)
    cdef int i
    erfinv_u = <double *> malloc(sizeof(double) * n)
    erfinv_v = <double *> malloc(sizeof(double) * n)
    sqrt = <double *> malloc(sizeof(double) * n)
    tmp_in = <double *> malloc(sizeof(double) * n)
    tmp_out = <double *> malloc(sizeof(double) * n)
    
    for i in xrange(n):
        tmp_in[i] = 2.0 * uu[i] - 1
    vdErfInv(<int> n, &tmp_in[0], &erfinv_u[0])
    for i in xrange(n):
        tmp_in[i] = 2.0 * vv[i] - 1
    vdErfInv(<int> n, &tmp_in[0], &erfinv_v[0])
    for i in xrange(n):
        tmp_in[i] = -rho[i] * rho[i] + 1
    vdSqrt(<int> n, &tmp_in[0], &sqrt[0])

    for i in prange(n, nogil=True):
        # this gives a nice picture in test_normal.test_conditional!
        # tmp_in[i] = rho[i] * erfinv_u[i] - erfinv_v[i] * sqrt[i]
        tmp_in[i] = (rho[i] * erfinv_u[i] - erfinv_v[i]) / sqrt[i]
    vdErf(<int> n, &tmp_in[0], &tmp_out[0])
    for i in xrange(n):
        result[i] = -.5 * tmp_out[i] + .5

    free(erfinv_u)
    free(erfinv_v)
    free(tmp_in)
    free(tmp_out)


cpdef norm_inv_cdf_given_u(double[:] uu,
                           double[:] qq,
                           double[:] rho,
                           double[:] result):
    cdef int n = len(uu)
    cdef int i
    erfinv_u = <double *> malloc(sizeof(double) * n)
    erfinv_v = <double *> malloc(sizeof(double) * n)
    sqrt = <double *> malloc(sizeof(double) * n)
    tmp_in = <double *> malloc(sizeof(double) * n)
    tmp_out = <double *> malloc(sizeof(double) * n)

    for i in xrange(n):
        tmp_in[i] = 2.0 * uu[i] - 1.0
    vdErfInv(<int> n, &tmp_in[0], &erfinv_u[0])
    for i in xrange(n):
        tmp_in[i] = -2.0 * (qq[i] - 0.5)
    vdErfInv(<int> n, &tmp_in[0], &erfinv_v[0])
    for i in xrange(n):
        tmp_in[i] = -rho[i] * rho[i] + 1
    vdSqrt(<int> n, &tmp_in[0], &sqrt[0])

    for i in prange(n, nogil=True):
        tmp_in[i] = rho[i] * erfinv_u[i] - erfinv_v[i] * sqrt[i]
    vdErf(<int> n, &tmp_in[0], &tmp_out[0])
    for i in xrange(n):
       result[i] = .5 * (1 + tmp_out[i])

    free(erfinv_u)
    free(erfinv_v)
    free(tmp_in)
    free(tmp_out)


cpdef erf(double[:] x,
          double[:] result):
    cdef int n = len(x)
    vdErf(<int> n, &x[0], &result[0])


cpdef erfinv(double[:] x,
             double[:] result):
    cdef int n = len(x)
    vdErfInv(<int> n, &x[0], &result[0])


