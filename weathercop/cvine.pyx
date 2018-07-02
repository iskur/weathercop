import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double[::1] _csim_one(int t,
                                  # double[::1] P,
                                  np.ndarray[DTYPE_t, ndim=1] P,
                                  cvine,
                                  double zero,
                                  double one):
    cdef double[::1] U = np.empty(len(P), dtype=float)
    cdef int d = cvine.d
    cdef int i, j
    cdef float q
    U[0] = P[0]
    U[1] = cvine[0, 1]["C^_1|0"](conditioned=P[None, 1],
                                 condition=P[None, 0],
                                 t=t)
    for j in range(2, d):
        q = P[j]
        for l in range(j - 1, -1, -1):
            cop = cvine[l, j][f"C^_{j}|{l}"]
            q = cop(conditioned=np.array([q]),
                    condition=P[None, l],
                    t=t)
            q = max(zero, min(one, q))
        U[j] = q
    return U


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double[::1] _cquant_one(int t,
                                    # double[::1] U,
                                    np.ndarray[DTYPE_t, ndim=1] U,
                                    cvine,
                                    double zero,
                                    double one):
    cdef double[::1] P = np.empty(len(U), dtype=float)
    cdef int d = cvine.d
    cdef int i, j
    cdef float q
    P[0] = U[0]
    P[1] = cvine[0, 1]["C_1|0"](conditioned=U[None, 1],
                                condition=np.asarray(P[None, 0]),
                                t=t)
    P[1] = max(zero, min(one, P[1]))
    for j in range(2, d):
        q = U[j]
        for l in range(j):
            cop = cvine[l, j][f"C_{j}|{l}"]
            q = cop(conditioned=np.array([q]),
                    condition=np.asarray(P[None, l]),
                    t=t)
        P[j] = q
    return P


cpdef csim_one(args):
    t, P, cvine, zero, one = args
    U = _csim_one(t, P.ravel(), cvine, zero, one)
    return np.array(U)


cpdef cquant_one(args):
    t, U, cvine, zero, one = args
    P = _cquant_one(t, U.ravel(), cvine, zero, one)
    return np.array(P)
