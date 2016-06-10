"""Various functions for time series anlysis.
If not specified differently, references are given on Luetkepohl "New
Introduction to Multiple Time Series Analysis".

.. currentmodule:: weathercop.ar_models

.. autosummary::
   :nosignatures:
   :toctree: generated/

   VAR_LS
   VAR_LS_sim
   VAREX_LS
   VAREX_LS_sim
   VAR_order_selection
   VAR_residuals
   VAREX_residuals
   VAR_LS_predict
   AIC
   FPE
   HQ
   SC
"""
import numpy as np
from scipy.linalg import kron
# import time_series as ts
# from lhglib.contrib import dirks_globals as my


def VAR_LS(data, p=2):
    """Least-Squares parameter estimation for a vector auto-regressive model of
    the form Y = B*Z + U. Records containing nans are excluded.
    Refer to the Least-Squares Estimator example 3.2.3 p.78. for method and
    variable names.

    Parameters
    ----------
    data : (K, T) ndarray
        K is the number of variables, T the number of timesteps
    p : int
        Autoregressive order of the process.

    Returns
    -------
    B : matrix
        Parameters of the fitted VAR-process.
    sigma_u: matrix
        Covariance matrix of the residuals.

    See also
    --------
    VAR_order_selection : Helps to find a p for parsimonious estimation.
    VAR_residuals : Returns the residuals based on given data and LS estimator
    VAR_LS_sim : Simulation based on LS estimator.
    VAR_LS_predict : Predict given prior data and LS estimator.
    """
    # number of variables
    if np.ndim(data) == 2:
        K, T = data.shape[0], data.shape[1] - p
    elif np.ndim(data) == 1:
        K, T = 1, len(data) - p
        data = data[np.newaxis, :]
    # Y is a (K, T) matrix
    Y = np.asmatrix(data[:, p:])

    Z = np.asmatrix(np.empty((K * p + 1, T)))
    Zt = np.empty((K * p + 1, 1))
    Zt[0] = 1
    for t in range(p, T + p):
        for subt in range(p):
            start_i = 1 + subt * K
            stop_i = 1 + (subt + 1) * K
            Zt[start_i:stop_i] = data[:, t - subt - 1].reshape((K, 1))
        Z[:, t - p] = Zt

    # delete all columns containing nans
    Y_nan_cols = np.where(np.isnan(Y))[1]
    Y = np.delete(Y, Y_nan_cols, axis=1)
    Z = np.delete(Z, Y_nan_cols, axis=1)
    Z_nan_cols = np.where(np.isnan(Z))[1]
    Y = np.delete(Y, Z_nan_cols, axis=1)
    Z = np.delete(Z, Z_nan_cols, axis=1)
    # no idea why there are remaining columns with nans in Z!
    Z_nan_cols = np.where(np.isnan(Z))[1]
    Y = np.asmatrix(np.delete(Y, Z_nan_cols, axis=1))
    Z = np.asmatrix(np.delete(Z, Z_nan_cols, axis=1))

    # B contains all the parameters we want: (nu, A1, ..., Ap)
    # Y = BZ + U
    B = Y * Z.T * (Z * Z.T).I

    # covariance matrix of the noise
    sigma_u = Y * Y.T - Y * Z.T * (Z * Z.T).I * Z * Y.T
    sigma_u /= T - K * p - 1

    return B, sigma_u


def VAREX_LS(data, p, ex):
    """Least-Squares parameter estimation for a vector auto-regressive model of
    the form

    ..math::y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + C x_t + u_t

    Records containing nans are excluded.
    Refer to the Least-Squares Estimator example 3.2.3 p.78. for method and
    variable names.

    Parameters
    ----------
    data : (K, T) ndarray
        K is the number of variables, T the number of timesteps
    ex : (T,) ndarray
        An external variable
    p : int
        Autoregressive order of the process.

    Returns
    -------
    B : matrix
        Parameters of the fitted VAR-process.
        B := (A_1, ..., A_p, C)
    sigma_u: matrix
        Covariance matrix of the residuals of the data.

    See also
    --------
    VAR_order_selection : Helps to find a p for parsimonious estimation.
    VAR_residuals : Returns the residuals based on given data and LS estimator
    VAR_LS_sim : Simulation based on LS estimator.
    VAR_LS_predict : Predict given prior data and LS estimator.
    """
    # number of variables
    if np.ndim(data) == 2:
        K, T = data.shape[0], data.shape[1] - p
    elif np.ndim(data) == 1:
        K, T = 1, len(data) - p
        data = data[np.newaxis, :]
    # Y is a (K, T) matrix
    Y = np.asmatrix(data[:, p:])

    Z = np.asmatrix(np.empty((K * p + 1, T)))
    Zt = np.empty((K * p, 1))
    Z[-1] = ex[p:].reshape(1, T)
    for t in range(p, T + p):
        for subt in range(p):
            start_i = subt * K
            stop_i = (subt + 1) * K
            Zt[start_i:stop_i] = data[:, t - subt - 1].reshape((K, 1))
        Z[:-1, t - p] = Zt

    # delete all columns containing nans
    Y_nan_cols = np.where(np.isnan(Y))[1]
    Y = np.delete(Y, Y_nan_cols, axis=1)
    Z = np.delete(Z, Y_nan_cols, axis=1)
    Z_nan_cols = np.where(np.isnan(Z))[1]
    Y = np.asmatrix(np.delete(Y, Z_nan_cols, axis=1))
    Z = np.asmatrix(np.delete(Z, Z_nan_cols, axis=1))

    # B contains all the parameters we want: (A1, ..., Ap, C)
    # Y = BZ + U
    B = Y * Z.T * (Z * Z.T).I

    # covariance matrix of the noise of data
    sigma_u = Y * Y.T - Y * Z.T * (Z * Z.T).I * Z * Y.T
    sigma_u /= T - K * p - 1

    return B, sigma_u


def SVAR_LS(data, doys, p=2, doy_width=30, fft_order=4, var_names=None,
            verbose=True):
    """Seasonal version of the least squares estimator."""
    K, T = data.shape
    Bs, sigma_us = [], []
    unique_doys = np.unique(doys)
    if verbose:
        progress_bar = my.ProgressBar(len(unique_doys))

    for doy_ii, doy in enumerate(unique_doys):
        ii = (doys > doy - doy_width) & (doys <= doy + doy_width)
        if (doy - doy_width) < 0:
            ii |= doys > (365. - doy_width + doy)
        if (doy + doy_width) > 365:
            ii |= doys < (doy + doy_width - 365.)
        B, sigma_u = VAR_LS(np.where(ii, data, np.nan), p=p)
        Bs += [B]
        # account for the smaller sample size
        sigma_us += [sigma_u * T / sum(ii)]
        if verbose:
            progress_bar.animate(int(doy_ii))

    Bs, sigma_us = np.asarray(Bs), np.asarray(sigma_us)
    if verbose:
        print()

    def matr_fft(M):
        return np.asarray(
            [[my.fourier_approx(M[:, ii, jj], fft_order)
              for jj in range(M.shape[2])] for ii in range(K)])

    return matr_fft(Bs), matr_fft(sigma_us)


def SVAR_LS_sim(Bs, sigma_us, doys, m=None, ia=None, m_trend=None, u=None,
                n_presim_steps=100, fixed_data=None):
    doys_ii = (doys % 365) / 365. * len(np.unique(doys))
    K = Bs.shape[0]
    p = (Bs.shape[1] - 1) // K
    Y = np.zeros((K, len(doys) + p))
    for date_i, doy_i in enumerate(doys_ii):
        Y[:, date_i + p] = \
            VAR_LS_sim(Bs[..., doy_i], sigma_us[..., doy_i], 1,
                       None if m is None else m[:, date_i, None],
                       None if ia is None else ia[:, date_i, None],
                       m_trend, n_presim_steps=0,
                       u=None if u is None else u[:, date_i, None],
                       prev_data=Y[:, date_i:date_i + p]).ravel()
    return Y[:, p:]


def VAR_LS_sim(B, sigma_u, T, m=None, ia=None, m_trend=None, u=None,
               n_presim_steps=100, fixed_data=None, prev_data=None):
    """Based on a least squares estimator, simulate a time-series of the form
    ..math::y(t) = nu + A1*y(t-1) + ... + Ap*y(t-p) + ut
    B contains (nu, A1, ..., Ap).
    See p. 707f

    Parameters
    ----------
    B : (K, K*p+1) matrix or ndarray
        Parameters of the VAR-process as returned from VAR_LS. K is the number
        of variables, p the autoregressive order.
    sigma_u : (K, K) matrix or ndarray
        Covariance matrix of the residuals as returned from VAR_LS.
    T : int
        Number of timesteps to simulate.
    m : (K,) ndarray, optional
        Process means (will be scaled according to B).
    ia : (K, T) ndarray, optional
        Interannual variability. Additional time-varying disturbance to the
        process means (will be scaled according to B).
    m_trend : (K,) ndarray, optional
        Change in means, that will be applied linearly so that this change is
        reached after the T timesteps.
    u : (K, T) ndarray, optional
        Residuals to be used instead of multivariate gaussian serially
        independent random numbers.
    n_presim_steps : int, optional
        Number of presimulation timesteps that will be thrown away.
    fixed_data : (K, T) ndarray, optional
        Data that will be fixed, i.e. at each timestep, these will be put in
        instead of the actual simulated values. Where fixed_data is nan, the
        simulated values will not be overwritten.

    Returns
    -------
    Y : (K, T) ndarray
        Simulated values.

    See also
    --------
    VAR_LS : Least-squares estimator (to get B and sigma_u).
    VAR_order_selection : Helps to find a p for parsimonious estimation.
    VAR_residuals : Returns the residuals based on given data and LS estimator
    VAR_LS_predict : Predict given prior data and LS estimator.
    """
    # number of variables
    K = B.shape[0]
    # order of VAR_LS-process
    p = (B.shape[1] - 1) // K
    n_sim_steps = T + p + n_presim_steps

    if m is not None:
        m = _scale_additive(m, B[:, 1:], p)
        # we have to expand m to include the pre-simulation timesteps
        m = np.concatenate((m[:, :n_presim_steps], m), axis=1)
    m_trend = np.asarray([0] * K) if m_trend is None else np.asarray(m_trend)
    m_trend = m_trend[:, np.newaxis]  # ha! erwischt!
    m_trend = _scale_additive(m_trend, B[:, 1:], p)

    if ia is not None:
        ia = _scale_additive(ia, B[:, 1:], p)

    # the first p columns are initial values, which will be omitted later
    Y = np.asmatrix(np.zeros((K, n_sim_steps)))
    if prev_data is not None:
        Y[:, :p] = np.asmatrix(prev_data[:, -p:])
        n_sim_steps -= n_presim_steps

    Ai = [np.asarray(B[:, 1 + i * K: 1 + (i + 1) * K]) for i in range(p)]

    if m is None and prev_data is None:
        nu = np.asmatrix(B[:, 0])
        # setting starting values to the process mean
        mu = np.asmatrix(np.identity(K))
        for i in range(p):
            mu -= Ai[i]
        mu = mu.I * nu
        Y[:, :p] = mu
    Y = np.asarray(Y)

    if u is None:
        u = np.random.multivariate_normal(K * [0], sigma_u, n_sim_steps - p)
        u = np.asmatrix(u.T)

    Y[:, -u.shape[1]:] += u
    if m is None and prev_data is None:
        Y[:, p:] += nu
    elif m is not None:
        Y[:, -m.shape[1]:] += m

    if ia is not None:
        Y[:, -ia.shape[1]:] += ia

    # apply changes as a trend
    Y[:, -T:] += np.arange(T, dtype=float) / T * m_trend

    start_t = n_sim_steps - T
    for t in range(p, n_sim_steps):
        for i in range(p):
            Y[:, t] += np.dot(Ai[i], Y[:, t - i - 1])

        if (fixed_data is not None) and (t >= start_t):
            # fixing what's asked to be held constant
            Y[:, t] = np.where(np.isnan(fixed_data[:, t - start_t]),
                               Y[:, t], fixed_data[:, t - start_t])

    return np.asarray(Y[:, -T:])


def VAREX_LS_sim(B, sigma_u, T, ex, m=None, ia=None, m_trend=None, u=None,
                 n_presim_steps=100, prev_data=None, ex_kwds=None):
    """Based on a least squares estimator, simulate a time-series of the form
    ..math::y(t) = A1*y(t-1) + ... + Ap*y(t-p) + C*x(t-1) + ut
    B contains (A1, ..., Ap, C).
    See p. 707f

    Parameters
    ----------
    B : (K, K*p+1) matrix or ndarray
        Parameters of the VAR-process as returned from VAR_LS. K is the number
        of variables, p the autoregressive order.
    sigma_u : (K, K) matrix or ndarray
        Covariance matrix of the residuals as returned from VAR_LS.
    ex : (T,) ndarray or function
        External variable. If given as a function, ex_t will be generated by
        calling ex(Y[:t], **ex_kwds), with Y being the simulated values.
    T : int
        Number of timesteps to simulate.
    m : (K,) ndarray, optional
        Process means (will be scaled according to B).
    ia : (K, T) ndarray, optional
        Interannual variability. Additional time-varying disturbance to the
        process means (will be scaled according to B).
    m_trend : (K,) ndarray, optional
        Change in means, that will be applied linearly so that this change is
        reached after the T timesteps.
    u : (K, T) ndarray, optional
        Residuals to be used instead of multivariate gaussian serially
        independent random numbers.
    n_presim_steps : int, optional
        Number of presimulation timesteps that will be thrown away.
    ex_kwds : dict, optional
        Keyword arguments to be passed to ex.

    Returns
    -------
    Y : (K, T) ndarray
        Simulated values.
    ex_out : (T,) ndarray
        External variable.

    See also
    --------
    VAR_LS : Least-squares estimator (to get B and sigma_u).
    VAR_order_selection : Helps to find a p for parsimonious estimation.
    VAR_residuals : Returns the residuals based on given data and LS estimator
    VAR_LS_predict : Predict given prior data and LS estimator.
    """
    # number of variables
    K = B.shape[0]
    # order of VAR_LS-process
    p = (B.shape[1] - 1) // K
    n_sim_steps = T + p + n_presim_steps

    try:
        len(ex)
        ex_isfunc = False
        ex_out = ex
    except TypeError:
        ex_kwds = {} if ex_kwds is None else ex_kwds
        ex_isfunc = True
        ex_out = np.empty(T)

    if m is not None:
        m = _scale_additive(m, B[:, 1:], p)
        # we have to expand m to include the pre-simulation timesteps
        m = np.concatenate((m[:, :n_presim_steps], m), axis=1)
    m_trend = np.asarray([0] * K) if m_trend is None else np.asarray(m_trend)
    m_trend = m_trend[:, np.newaxis]  # ha! erwischt!
    m_trend = _scale_additive(m_trend, B[:, 1:], p)

    if ia is not None:
        ia = _scale_additive(ia, B[:, 1:], p)

    # the first p columns are initial values, which will be omitted later
    Y = np.asmatrix(np.zeros((K, n_sim_steps)))
    if prev_data is not None:
        Y[:, :p] = np.asmatrix(prev_data[:, -p:])
        n_sim_steps -= n_presim_steps

    Ai = [np.asarray(B[:, i * K: (i + 1) * K]) for i in range(p)]
    C = np.asarray(B[:, -1])

    # if m is None and prev_data is None:
    #     nu = np.asmatrix(B[:, 0])
    #     # setting starting values to the process mean
    #     mu = np.asmatrix(np.identity(K))
    #     for i in range(p):
    #         mu -= Ai[i]
    #     mu = mu.I * nu
    #     Y[:, :p] = mu
    Y = np.asarray(Y)

    if u is None:
        u = np.random.multivariate_normal(K * [0], sigma_u, n_sim_steps - p)
        u = np.asmatrix(u.T)

    Y[:, -u.shape[1]:] += u
    # if m is None and prev_data is None:
    #     Y[:, p:] += nu
    if m is not None:
        Y[:, -m.shape[1]:] += m

    if ia is not None:
        Y[:, -ia.shape[1]:] += ia

    # apply changes as a trend
    Y[:, -T:] += np.arange(T, dtype=float) / T * m_trend

    start_t = n_sim_steps - T
    for t in range(p, n_sim_steps):
        for i in range(p):
            Y[:, t] += np.dot(Ai[i], Y[:, t - i - 1])

        if t >= start_t:
            if ex_isfunc:
                ex_t = ex(Y[:, :t], **ex_kwds)
                ex_out[t - p - start_t] = ex_t
            else:
                ex_t = ex[t - p - start_t]
            Y[:, t] += np.squeeze(C * ex_t)

    return np.asarray(Y[:, -T:]), ex_out


def VAR_residuals(data, B, p=2):
    K, T = data.shape

    # were we given a B with the nus (in the first column)?
    if B.shape[1] % (K * p) == 1:
        mean_adjusted = False
        i_shift = 1
        # what is the process mean?
        nu = np.asmatrix(B[:, 0])
        mu = np.asmatrix(np.identity(K))
        for i in range(p):
            Ai = B[:, 1 + i * K: 1 + (i + 1) * K]
            mu -= Ai
        mu = mu.I * nu
        nu = np.asarray(nu).ravel()
    else:
        mean_adjusted = True
        i_shift = 0
        # estimate the process means from the data.
        mu = np.asmatrix(np.mean(data, axis=1)[:, np.newaxis])

    # set the pre-sample period to the process means
    data = np.concatenate((np.empty((K, p)), data), axis=1)
    data[:, :p] = mu
    resi = np.copy(data)

    for t in range(p, T + p):
        for i in range(p):
            Ai = np.asarray(B[:, i_shift + i * K: i_shift + (i + 1) * K])
            resi[:, t] -= np.dot(Ai, data[:, t - i - 1])
        if not mean_adjusted:
            resi[:, t] -= nu
    return np.asarray(resi[:, p:] + mu) if mean_adjusted else resi[:, p:]


def VAREX_residuals(data, ex, B, p=2, ex_kwds=None):
    K, T = data.shape

    try:
        len(ex)
        ex_isfunc = False
    except TypeError:
        ex_kwds = {} if ex_kwds is None else ex_kwds
        ex_isfunc = True

    # set the pre-sample period to the process means
    data = np.concatenate((np.zeros((K, p)), data), axis=1)
    resi = np.copy(data)

    C = np.asarray(B[:, -1])
    for t in range(p, T + p):
        for i in range(p):
            Ai = np.asarray(B[:, i * K: (i + 1) * K])
            resi[:, t] -= np.dot(Ai, data[:, t - i - 1])

        if ex_isfunc:
            ex_t = ex(data[:, :t], **ex_kwds)
        else:
            ex_t = ex[t - p]

        resi[:, t] -= np.squeeze(C * ex_t)
    return np.asarray(resi[:, p:])


def SVAR_residuals(data, doys, B, p=2):
    K, T = data.shape
    doys_ii = (doys % 365) / 365. * len(np.unique(doys))

    # were we given a B with the nus (in the first column)?
    if B.shape[1] % (K * p) == 1:
        mean_adjusted = False
        i_shift = 1
    else:
        mean_adjusted = True
        i_shift = 0
        # estimate the process means from the data.
        mu = np.asmatrix(np.mean(data, axis=1)[:, np.newaxis])

    def process_mean(B):
        nu = np.asmatrix(B[:, 0]).T
        mu = np.asmatrix(np.identity(K))
        for i in range(p):
            Ai = B[:, 1 + i * K: 1 + (i + 1) * K]
            mu -= Ai
        mu = mu.I * nu
        return np.asarray(nu).ravel(), mu

    # set the pre-sample period to the process means
    data = np.concatenate((np.empty((K, p)), data), axis=1)
    data[:, :p] = process_mean(B[..., 0])[1]
    resi = np.copy(data)

    for t in range(p, T + p):
        for i in range(p):
            Ai = B[:, i_shift + i * K: i_shift + (i + 1) * K, doys_ii[t - p]]
            resi[:, t] -= \
                np.squeeze(np.asarray(np.asmatrix(Ai) *
                                      data[:, t - i - 1].reshape(K, 1)))
#        if not mean_adjusted:
#            resi[:, t] -= process_mean(B[..., doys_ii[t - p]])[0]
    return np.asarray(resi[:, p:] + mu) if mean_adjusted else resi[:, p:]


def VARMA_LS_prelim(data, p, q):
    """Preliminary version of the general simple least-squares vector
    autoregressive estimator for a vector autoregressive moving-average
    process. See p.474ff"""
    K, T = data.shape[0], data.shape[1] - p
    # number of parameters
    N = K ** 2 * (p + q)

    # first estimate ut by calculating the residuals from a long VAR-process
    B = VAR_LS(data, max(10, int(1.5 * (p + q))))[0]
    ut_est = VAR_residuals(data, B, p)

    Y = np.asmatrix(data[:, p:])
    X = np.asmatrix(np.ones((K * (p + q), T)))
    Xt = np.zeros((K * (p + q), 1))
    for t in range(p, T + p):
        for subt in range(p):
            start_i = subt * K
            stop_i = (subt + 1) * K
            Xt[start_i:stop_i] = data[:, t - subt - 1].reshape((K, 1))
        for subt in range(p, p + q):
            start_i = subt * K
            stop_i = (subt + 1) * K
            Xt[start_i:stop_i] = ut_est[:, t - subt - p - 1].reshape((K, 1))
        X[:, t - p] = Xt

    # R might not be necessary since we do not limit any parameters here
    R = np.asmatrix(np.identity(N))
    IK = np.asmatrix(np.identity(K))
    gamma = ((R.T * kron(X * X.T, IK) * R).I * R.T * kron(X, IK) * vec(Y))
#    gamma = (X * sp.linalg.kron(X.T, IK)).I * sp.linalg.kron(X, IK) * vec(Y)
    residuals_arma_vec = vec(Y) - kron(X.T, IK) * R * gamma
    residuals_arma = residuals_arma_vec.reshape((K, T), order="F")
    # make the residuals have the same length as the data
    residuals_arma = np.concatenate((np.zeros((K, p)), residuals_arma),
                                    axis=1)
    sigma_u_arma = residuals_arma * residuals_arma.T / T
    AM = gamma.reshape((K, -1), order="F")
    # the following expression leads to the same result...
#    AM = Y * X.T * (X * X.T).I
    return AM, sigma_u_arma, residuals_arma


def VARMA_LS_sim(AM, p, q, sigma_u, means, T, S=None, m=None, ia=None,
                 m_trend=None, n_sim_multiple=2, fixed_data=None):
    """Generates a time series based on the VARMA-parameters AM.
    S and m should be sequences of length K. S is a variable-discerning
    multiplier and m a adder, respectively.

    Parameters
    ----------
    AM :       (K,K*(p+q)) matrix
               The parameters of the VARMA-process. The first p columns are
               interpreted as the Ai-matrices of the auto regressive part. The
               last q columns as the Mi-matrices of the moving average part. K
               is the number of variables simulated.
    p :        integer
               Order of the auto regressive process.
    q :        integer
               Order of the moving average process.
    sigma_u :  (K,K) matrix
               Covariance matrix of the residuals.
    means :    (K,) matrix
               Process means. Used as starting values.
    T :        integer
               Desired length of the output time series.
    S :        (K,K) matrix, optional
               Used as multiplicative change of the disturbance vector to
               increase the variance of the output.
    m :        (K,T) array_like, optional
               Used as additive change during simulation to increase mean of
               the output.
    n_sim_multiple : integer
                    Generate n_sim_multiple * T timesteps. Only the last T
                    timesteps will be returned.
    ia :       (K,T) array_like, optional
               Interannual variability. Used as an additive change during
               simulation to get time-dependent disturbances.
    m_trend :  (K,) array_like, optional
               Used as additive change gradient during simulation to increase
               mean of the output gradually.
    fixed_data : (K,T) array_like, optional
                Keeps the provided time-series fixed. Use np.nans to signify
                values that are not fixed.
                Can be used to simulate hierarchically.

    Returns
    -------
    out :    (K, T) ndarray
             K-dimensional simulated time series.

    """
    K = AM.shape[0]
    if S is None:
        S = np.asmatrix(np.identity(K, dtype=float))
    n_sim_steps = n_sim_multiple * T + p
    if m is None:
        m = np.zeros((K, n_sim_steps))
    else:
        m = _scale_additive(m, AM, p)
        # we have to expand m to include the pre-simulation timesteps
        m = np.tile(m, n_sim_multiple)
    m_trend = np.asarray([0] * K) if m_trend is None else np.asarray(m_trend)
    m_trend = m_trend[:, np.newaxis]  # ha! erwischt!
    m_trend = _scale_additive(m_trend, AM, p)

    if ia is not None:
        ia = _scale_additive(ia, AM, p)

    # the first p columns are initial values, which will be omitted later
    Y = np.asmatrix(np.zeros((K, n_sim_steps)))
    Y[:, :p] = means.reshape((K, -1))

    Y[:, -m.shape[1]:] += m

    start_t = Y.shape[1] - T

    ut = np.asmatrix([np.random.multivariate_normal(K * [0], sigma_u)
                      for i in range(q)]).reshape((K, q))

    for t in range(p, n_sim_steps):
        # shift the old values back and draw a new random vector
        ut[:, :-1] = ut[:, 1:]
        ut[:, -1] = \
            np.random.multivariate_normal(K * [0], sigma_u).reshape(K, 1)
        Y[:, t] = ut[:, -1][np.newaxis, :]

        # non-standard scenario stuff
        # beware of matrix multiplication! *= is not what we want
        Y[:, t] = S * Y[:, t]
        if t > start_t:
            if ia is not None:
                Y[:, t] += np.asmatrix(ia[:, t - start_t]).T
            # apply changes as a trend
            Y[:, t] += float(t - start_t) / T * m_trend

        # conventional VARMA things
        for i in range(p):
            Ai = AM[:, i * K: (i + 1) * K]
            Y[:, t] += Ai * Y[:, t - i - 1]
        for i in range(p, p + q):
            Mi = AM[:, i * K: (i + 1) * K]
            Y[:, t] += Mi * ut[:, -1 - i + p]

        if (fixed_data is not None) and (t >= start_t):
            # fixing what's asked to be held constant
            Y[:, t] = \
                np.where(np.isnan(fixed_data[:, t - start_t, np.newaxis]),
                         Y[:, t], fixed_data[:, t - start_t, np.newaxis])

        if fixed_data is None:
            Y[:, t] += means.reshape((K, -1))

    return np.asarray(Y[:, -T:])


def vec(A):
    """The vec operator stacks 2dim matrices into 1dim vectors column-wise.
    See p.661f.

    >>> A = np.matrix(np.arange(4).reshape(2, 2))
    >>> A
    matrix([[0, 1],
            [2, 3]])
    >>> vec(A)
    matrix([[0],
            [2],
            [1],
            [3]])
    """
    return np.asmatrix(A.T.ravel()).T


def vech(A):
    """The vech operator removes the upper triangular part of a matrix and
    returns the rest in a column-stacked form.
    See p.661f.

    >>> A = np.matrix(np.arange(4).reshape(2, 2))
    >>> A
    matrix([[0, 1],
            [2, 3]])
    >>> vech(A)
    matrix([[0],
            [2],
            [3]])
    """
    rows, columns = np.mgrid[0:A.shape[0], 0:A.shape[1]]
    return np.asmatrix(A.T[rows.T >= columns.T]).T


def SC(sigma_u, p, T):
    """Schwarz criterion for VAR_LS order selection (p.150). To be minimized.
    """
    K = sigma_u.shape[0]
    return np.log(np.linalg.det(sigma_u)) + np.log(T) / T * p * K ** 2


def HQ(sigma_u, p, T):
    """Hannan-Quinn for VAR_LS order selection (p.150). To be minimized.
    """
    K = sigma_u.shape[0]
    return np.log(np.linalg.det(sigma_u)) + np.log(np.log(T)) / T * p * K ** 2


def AIC(sigma_u, p, T):
    """Akaike Information Criterion for order selection of a VAR process.
    See p.147"""
    K = sigma_u.shape[0]
    return np.log(np.linalg.det(sigma_u)) + (2 * p * K ** 2) / T


def FPE(sigma_u, p, T):
    """Final prediction error.
    See p.147"""
    K = sigma_u.shape[0]
    return (((T + p * K + 1) / (T - p * K - 1)) ** K * np.linalg.det(sigma_u))


def VAR_order_selection(data, p_max=10, criterion=SC, estimator=VAR_LS,
                        est_kwds=None):
    """Order selection for VAR processes to allow parsimonious
    parameterization.

    Parameters
    ----------
    data : (K, T) ndarray
        Input data with K variables and T timesteps.
    p_max : int, optional
        Maximum number of autoregressive order to evaluate.
    criterion : function, optional
        Information criterion that accepts sigma_u, p, and T and returns
        something that gives small values for a parsimonious set of these
        parameters.

    Returns
    -------
    p : int
        Suggested autoregressive order.

    See also
    --------
    AIC : Akaike Information criterion
    FPE : Final Prediction Error
    HQ : Hannan-Quinn information criterion
    SC : Schwartz Criterion
    VAR_LS : Least-squares estimator.
    VAR_residuals : Returns the residuals based on given data and LS estimator
    VAR_LS_sim : Simulation based on LS estimator.
    VAR_LS_predict : Predict given prior data and LS estimator.

    """
    T = data.shape[1]
    if est_kwds is None:
        est_kwds = {}
    return np.argmin([criterion(estimator(data, p, **est_kwds)[1], p, T)
                      for p in range(p_max + 1)])


def VARMA_order_selection(data, p_max=5, q_max=5, criterion=SC,
                          plot_table=False):
    """Returns p and q, the orders of a VARMA process that allows for
    parsimonious parameterization.
    Naive extension of VAR_order_selection without a theoretical basis!"""
    K, T = data.shape
    sigma_us = np.nan * np.empty((p_max + 1, q_max + 1, K, K))
    # we ignore the cases where either p or q is 0, because VARMA_LS_prelim
    # chokes on that
    for p in range(1, p_max + 1):
        for q in range(q_max + 1):
            if q is 0:
                sigma_us[p, q] = VAR_LS(data, p)[1]
            sigma_us[p, q] = VARMA_LS_prelim(data, p, q)[1]

    crits = list(criterion)
    criterion_table = np.nan * np.empty((len(crits), p_max + 1, q_max + 1))
    for crit_i, crit in enumerate(crits):
        for p in range(1, p_max + 1):
            for q in range(q_max + 1):
                criterion_table[crit_i, p, q] = crit(sigma_us[p, q], p + q, T)

    if plot_table:
        for crit_i, crit in enumerate(crits):
            ts.matr_img(criterion_table[crit_i],
                        "Information criterion table. %s" % repr(crit))
            ts.plt.xlabel("q")
            ts.plt.ylabel("p")

    p_mins, q_mins = list(
        zip(*[np.unravel_index(np.nanargmin(criterion_table[ii]),
                               criterion_table[ii].shape)
              for ii in range(len(crits))]))
    return p_mins, q_mins, criterion_table


def _scale_additive(additive, A, p=None):
    """Scale an additive online component of a simulation. This prevents
    the overshooting due to auto- and crosscorrelations.

    Parameters
    ----------
    additive :   (K,) or (K,T) matrix or ndarray
                 Additive component to be scaled. K is the number of variables
                 simulated.Can be m, m_trend or ia of VARMA_LS_sim, for
                 example.
    A :          (K,K*p) matrix or ndarray
                 Parameters of the VAR process. p is the order of the VAR
                 process.
    p :          int, optional
                 Order of the VAR process. If given, only the first K*p columns
                 of A will be interpreted as the parameters of the VAR process.
                 Allows AM to be given as A, which also includes the VMA
                 parameters.

    Returns
    -------
    additive :   (K,) or (K,T) ndarray
                 Scaled additive component.
    """

    A = np.asarray(A)  # np.asmatrix(A)
    K = A.shape[0]
    if p is None:
        p = A.shape[1] / K
        if p * K != A.shape[1]:
            raise ValueError("Matrix A is not (K,K*p)-shape.")

    # assure that the Ai and the additive is aligned
    additive = np.asarray(additive)
    additive = additive.reshape((K, -1))

    scale_matrix = np.identity(K)
    for i in range(p):
        scale_matrix -= A[:, i * K: (i + 1) * K]

    # the following is equivalent to:
    #    # number of timesteps
    #    T = additive.shape[1]
    #    scaled_additive = np.zeros_like(additive).astype(float)
    #        for t in xrange(T):
    #            scaled_additive[:, t] = scale_matrix * additive[:, t]
    # (considering everything is of type matrix)
    scaled_additive = np.sum(scale_matrix[..., np.newaxis] *
                             additive[np.newaxis, ...], axis=1)

    return np.asarray(scaled_additive)


###############################################################################
## WARNING! The following functions were NOT tested thoroughly!!!!!!!!!!!!!!!!!
###############################################################################

def VAR_LS_predict(data_past, B, sigma_u, T=1, n_realizations=1):
    """Based on a least squares estimator, predict a time-series of the form
    ..math::y(t) = nu + A1*y(t-1) + ... + Ap*y(t-p) + ut
    B contains (nu, A1, ..., Ap).

    Parameters
    ----------
    data_past :      (K, p) ndarray
    B :              (K, p * K + 1) ndarray
                     Parameters of the VAR-process of order p.
    sigma_u :        (K, K) ndarray
                     Covariance matrix of the residuals of the VAR-process.
    T :              int
                     Number of timesteps to predict.
    n_realizations : int
                     Number of realizations. If > 1, gaussian disturbances
                     are added. So if n_realizations=1, the prediction is a
                     best guess.

    Returns
    -------
    Y :             (K, T) or (K, T, n_realizations) ndarray

    References
    ----------
    See p. 707f"""
    # number of variables
    K = B.shape[0]
    # order of VAR_LS-process
    p = (B.shape[1] - 1) / K
    nu = np.asmatrix(B[:, 0]).ravel()

    # the first p columns are initial values, which will be omitted later
    Y = np.zeros((K, data_past.shape[1] + T, n_realizations))
    Y[:, :-T] = data_past[..., np.newaxis]

    for t in range(Y.shape[1] - T, Y.shape[1]):
        for r in range(n_realizations):
            Y[:, t, r] = nu
            if n_realizations > 1:
                Y[:, t, r] += np.random.multivariate_normal(K * [0], sigma_u)

            for i in range(p):
                Ai = np.asarray(B[:, 1 + i * K: 1 + (i + 1) * K])
                Y[:, t, r] += np.squeeze(np.dot(Ai, Y[:, t - i - 1, r]))

    return np.squeeze(Y[:, -T:])


def VAR_YW(data, p=2):
    """Yule-Walker parameter estimation for a vector auto-regressive model of
    the form Y^0 = A*X + U
    Refer to p. 83ff.
    Here we assume that the data is already mean-adjusted!
    """
    # number of variables
    K, T = data.shape[0], data.shape[1] - p
    # Y is a (K, T) matrix
    Y = np.asmatrix(data[:, p:])

    # X is nearly the same as Z in VAR_LS, but without the first row of ones
    X = np.asmatrix(np.empty((K * p, T)))
    Xt = np.empty((K * p, 1))
    for t in range(p, T + p):
        for subt in range(p):
            # HACK! check the p
            Xt[subt * K:(subt + 1) * K] = data[:, t - subt - 1].reshape((K, 1))
        X[:, t - p] = Xt

    # A contains all the parameters (A1, ..., Ap)
    A = np.asmatrix(np.empty((K, K * p)))
    Gamma_y = np.asmatrix(np.empty_like(A))
    # cov-matrices for up to p lags
    for lag in range(1, p + 1):
        # unbiased cross-covariance
        cov = ts.cross_cov(data, lag) / (T + p - lag)
        start_i = (lag - 1) * cov.shape[0]
        stop_i = lag * cov.shape[0]
        Gamma_y[:, start_i:stop_i] = cov
    Gamma_Y = np.asmatrix(np.empty((K * p, K * p)))
    for ii in range(p):
        start_i = ii * K
        stop_i = (ii + 1) * K
        for jj in range(p):
            start_j = jj * K
            stop_j = (jj + 1) * K
            cov = ts.cross_cov(data, ii - jj) / (T + p - abs(ii - jj))
            Gamma_Y[start_i:stop_i, start_j:stop_j] = cov
    A = Gamma_y * Gamma_Y.I

    # lets use the same noise as VAR_LS. no idea if this is justified
    sigma_u = Y * Y.T - Y * X.T * (X * X.T).I * X * Y.T
    sigma_u /= T - K * p - 1

#    A_dash = Y * X.T * (X * X.T).I
#    matr_img(np.asarray(A), "A")
#    matr_img(np.asarray(A_dash), "A_dash")
#    plt.show()
    return A, sigma_u


def VAR_YW_sim(A, sigma_u, T):
    """Based on a Yule-Walker estimator, simulate a time-series of the form
    ..math:: y(t) = A_1*y(t-1) + ... + A_p*y(t-p) + u(t)
    A contains (A1, ..., Ap).
    See p. 707f"""
    # number of variables
    K = A.shape[0]
    # order of VAR-process
    p = A.shape[1] / K

    # the first p columns are initial values, which will be omitted later
    Y = np.asmatrix(np.zeros((K, T + p)))

    for t in range(p, T + p):
        ut = np.random.multivariate_normal(K * [0], sigma_u).reshape(K, 1)
        Y[:, t] = ut
        for i in range(p):
            Ai = A[:, i * K: (i + 1) * K]
            Y[:, t] += Ai * Y[:, t - i]

    return np.asarray(Y[:, p:])


# def VAR_YW_residuals(data, A, p=2):
#    K, T = data.shape[0], data.shape[1] - p
#    resi = np.copy(data)
#    for t in xrange(p, T + p):
#        for i in range(p):
#            Ai = A[:, i * K: (i + 1) * K]
#            resi[:, t] -= \
#                np.squeeze(np.asarray(Ai * data[:, t - i - 1].reshape(K, 1)))
#    return resi


def _ut_gamma_part(data, p, q, AM, ut):
    """Recursive calculation of the partial derivatives del ut /del gamma.
    See Lemma 12.1 p.468"""
    K, T = data.shape
    N = K ** 2 * (p + q)

    Y = np.asmatrix(data[:, p:])
    R = np.asmatrix(np.ones((N, N)))  # np.identity(N))
    A_0 = np.asmatrix(np.identity(K))
    IK_zeros = np.asmatrix(np.zeros((K ** 2, N)))
    IK_zeros[:K ** 2, :K ** 2] = np.asmatrix(np.identity(K ** 2))
#    zero_IK = np.asmatrix(np.zeros((N, N + 1)))
#    zero_IK[:, 1:] = np.asmatrix(np.identity(N))
    zero_IK = np.asmatrix(np.identity(N))
    ut_gamma_part = np.zeros((K, N, T + p))

    for t in range(p, T - p):
        varma = np.asmatrix(np.zeros((K, 1)))
        for i in range(p):
            Ai = AM[:, i * K: (i + 1) * K]
            varma += Ai * Y[:, t - i]
        for i in range(p, p + q):
            Mi = AM[:, i * K: (i + 1) * K]
            varma += Mi * ut[:, -1 - i + p, np.newaxis]

        prev_yu = np.empty(K * (p + q))
        for i in range(p):
            prev_yu[i * K:(i + 1) * K] = Y[:, t - i].T
        for i in range(p, p + q):
            prev_yu[i * K:(i + 1) * K] = ut[:, t - i].T
        prev_yu = np.asmatrix(prev_yu)
        M_gamma_part = np.asmatrix(np.zeros((K, N)))
        for i in range(p, p + q):
            Mi = AM[:, i * K: (i + 1) * K]
            M_gamma_part += Mi * np.asmatrix(ut_gamma_part[..., t - i + p])
        ut_gamma_part[..., t] = \
            ((A_0.I * kron(varma.T, A_0.T)) * IK_zeros * R -
             kron(prev_yu, A_0.I) * zero_IK * R -
             A_0.I * M_gamma_part)
    return ut_gamma_part


def VARMA_LS(data, p, q, rel_change=1e-3):
    """Implementation of the scoring algorithm to fit a VARMA model. p.470ff"""
    AM_pre, sigma_u_pre = VARMA_LS_prelim(data, p, q)[:2]
    # do not trust the estimator of the residuals
#    ut = residuals_pre
    ut = np.matrix(VARMA_residuals(data, AM_pre, p, q))
    K, T = data.shape
    N = K ** 2 * (p + q)

    det_new = np.linalg.det(sigma_u_pre)
    # set det_old to something that will cause the while loop to execute at
    # least one time
    det_old = rel_change ** -1 * det_new
    print(("Determinant of preliminary residual covariance matrix: %f" %
           det_old))

    AM = AM_pre
    gamma = vec(AM)

    ts.matr_img(AM, "AM p=%d q=%d Preliminary" % (p, q))
    ii = 0
    while ((det_new > 1e-60) and
           (np.abs(det_old - det_new) / det_old > rel_change)):
        ut_gamma_part = _ut_gamma_part(data, p, q, AM, ut)
        sigma_u_gamma = T ** -1 * np.sum([ut[:, t] * ut[:, t].T  # np.newaxis]
                                          for t in range(ut.shape[1])],
                                         axis=0)
        sigma_u_gamma = np.asmatrix(sigma_u_gamma)
        sigma_u_gamma_inv = sigma_u_gamma.I
        # information matrix
        IM = np.sum([ut_gamma_part[..., t].T * sigma_u_gamma_inv *
                     ut_gamma_part[..., t]
                     for t in range(T)], axis=0)
        IM = np.asmatrix(IM)
#        likeli_gamma_part = np.sum([ut[:, t].T * #, np.newaxis].T *
        likeli_gamma_part = np.sum([ut[:, t, np.newaxis].T *
                                    sigma_u_gamma_inv * ut_gamma_part[..., t]
                                    for t in range(ut.shape[1])],
                                   axis=0)
        gamma -= IM.I * likeli_gamma_part.T

        AM = gamma.reshape((K, K * (p + q)), order="F")
        ut = np.matrix(VARMA_residuals(data, AM, p, q))
        Y = np.asmatrix(data)  # [:, p:])
        X = np.asmatrix(np.ones((K * (p + q), T)))
        Xt = np.zeros((K * (p + q), 1))
        for t in range(p, T):
            for subt in range(p):
                start_i = subt * K
                stop_i = (subt + 1) * K
                Xt[start_i:stop_i] = data[:, t - subt - 1].reshape((K, 1))
            for subt in range(p, p + q):
                start_i = subt * K
                stop_i = (subt + 1) * K
                Xt[start_i:stop_i] = ut[:, t - subt - p - 1].reshape((K, 1))
            X[:, t - p] = Xt
        IK = np.asmatrix(np.identity(K))
        R = np.asmatrix(np.identity(N))
        gamma2 = ((R.T * kron(X * X.T, IK) * R).I * R.T * kron(X, IK) * vec(Y))
        residuals_arma_vec = vec(Y) - kron(X.T, IK) * R * gamma2
        residuals_arma = residuals_arma_vec.reshape((K, T), order="F")
        # make the residuals have the same length as the data
        ut = np.concatenate((np.zeros((K, p)), residuals_arma), axis=1)

#        sigma_u = np.cov(ut)
        sigma_u = residuals_arma * residuals_arma.T / T

        det_new, det_old = np.linalg.det(sigma_u), det_new
        print("Determinant of residual covariance matrix: %f" % det_new)
        ii += 1
        if ii > 1:
            ts.matr_img(np.asarray(gamma2.reshape((K, K * (p + q)),
                                                  order="F")),
                        "AM p=%d q=%d Iteration: %d" % (p, q, ii))

    AM = gamma2.reshape((K, K * (p + q)), order="F")
    return AM, sigma_u, ut


def VARMA_residuals(data, AM, p, q):
    K, T = data.shape[0], data.shape[1] - p
    resi = np.copy(data)
    for t in range(p, T + p):
        for i in range(p):
            Ai = AM[:, i * K: (i + 1) * K]
            resi[:, t] -= \
                np.squeeze(np.asarray(Ai * data[:, t - i - 1].reshape(K, -1)))
        for i in range(p, p + q):
            Mi = AM[:, i * K: (i + 1) * K]
            resi[:, t] -= \
                np.squeeze(np.asarray(Mi *
                                      resi[:, t - i + p - 1].reshape(K, -1)))
    return resi


if __name__ == "__main__":
    import doctest
    doctest.testmod()
