# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float_t DTYPE_t
# ctypedef np.int_t DTYPE_i
ctypedef np.int64_t DTYPE_i

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double[:, ::1] _csim(np.ndarray[DTYPE_t, ndim=2] P,
                                 cvine,
                                 double zero,
                                 double one,
                                 # int T_sim
                                 np.ndarray[DTYPE_i, ndim=1] tt,
                                 unsigned int stop_at,
                                 ):
    cdef np.ndarray[DTYPE_t, ndim=2] U = np.empty_like(P,
                                                       dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] Q = np.empty((len(tt),),
                                                  dtype=np.float64)
    # cdef np.ndarray[unsigned int, ndim=1] tt = np.arange(T_sim, dtype=np.uint8)
    # cdef np.ndarray[int, ndim=1] tt = np.arange(T_sim, dtype=np.int32)
    # cdef unsigned int d = cvine.d
    stop_at = min(stop_at, cvine.d)
    cdef unsigned int j, l, t
    U[0] = P[0]
    if stop_at == 0:
        return U
    U[1] = cvine[0, 1]["C^_1|0"](conditioned=P[1],
                                 condition=P[0],
                                 t=tt)
    for j in range(2, stop_at):
        Q = P[j]
        for l in range(j - 1, -1, -1):
            cop = cvine[l, j][f"C^_{j}|{l}"]
            Q = cop(conditioned=Q,
                    condition=P[l],
                    t=tt)
            for t in tt:
                if Q[t] < zero:
                    Q[t] = zero
                elif Q[t] > one:
                    Q[t] = one
        U[j] = Q
    return U


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double[:, ::1] _cquant(np.ndarray[DTYPE_t, ndim=2] U,
                                   cvine,
                                   double zero,
                                   double one,
                                   np.ndarray[DTYPE_i, ndim=1] tt,
                                   # int T
                                   ):
    cdef np.ndarray[DTYPE_t, ndim=2] P = np.empty_like(U, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] Q = np.empty((len(tt),),
                                                  dtype=np.float64)
    # cdef np.ndarray[unsigned int, ndim=1] tt = np.arange(T, dtype=np.uint32)
    # cdef np.ndarray[int, ndim=1] tt = np.arange(T, dtype=np.int32)
    cdef unsigned int d = cvine.d
    cdef unsigned int i, j, t
    P[0] = U[0]
    P[1] = cvine[0, 1]["C_1|0"](conditioned=U[1],
                                condition=np.asarray(P[0]),
                                t=tt)
    for t in tt:
        if P[1, t] < zero:
            P[1, t] = zero
        elif P[1, t] > one:
            P[1, t] = one
    for j in range(2, d):
        Q = U[j]
        for l in range(j):
            cop = cvine[l, j][f"C_{j}|{l}"]
            Q = cop(conditioned=Q,
                    condition=np.asarray(P[l]),
                    t=tt)
            for t in tt:
                if Q[t] < zero:
                    Q[t] = zero
                elif Q[t] > one:
                    Q[t] = one
        P[j] = Q
    return P


cpdef csim(args):
    P, cvine, zero, one, tt, stop_at = args
    if stop_at is None:
        stop_at = cvine.d
    U = _csim(P, cvine, zero, one, tt, stop_at)
    if stop_at < cvine.d:
        U[stop_at + 1:] = np.nan
    return np.array(U)

cpdef cquant(args):
    U, cvine, zero, one, tt = args
    P = _cquant(U, cvine, zero, one, tt)
    return np.array(P)
