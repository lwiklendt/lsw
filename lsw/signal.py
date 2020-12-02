import collections
import numpy as np
import scipy.ndimage
import scipy.signal
from numba import njit
from scipy.special import logit, expit


def make_nondecreasing(x):
    dx = np.diff(x)
    dx[dx < 0] = 0
    return x[0] + np.r_[0, np.cumsum(dx)]


def ceil_pow_2(k):
    if bin(k).count('1') > 1:
        k = 1 << k.bit_length()
    return k


def box_filter(x, size, k=1, **kwargs):
    size = int(np.round(size))
    y = x.copy()
    for _ in range(k):
        scipy.ndimage.uniform_filter1d(y, size, output=y, **kwargs)
    return y


def box_bandpass(x, low_cut_size, high_cut_size, k=1, **kwargs):
    y = box_filter(x, high_cut_size, k=k, **kwargs)
    return y - box_filter(y, low_cut_size, k=k, **kwargs)


def box_baseline(x, size, k=1, iters=10, **kwargs):
    orig = x.copy()
    for i in range(iters):
        x = box_filter(x, size, k=k, **kwargs)
        if i < iters - 1:
            x = np.minimum(orig, x)
    return x


def baseline_gauss(x, sigma, iters):
    orig = x
    for i in range(iters):
        x = scipy.ndimage.gaussian_filter1d(x, sigma)
        if i < iters - 1:
            x = np.minimum(orig, x)
    return x


def butter_bandpass(x, fs_hz, low_hz, high_hz, order=15, axis=-1):
    nyq_hz = 0.5 * fs_hz
    sos = scipy.signal.butter(order, [low_hz / nyq_hz, high_hz / nyq_hz], btype='bandpass', output='sos')
    return scipy.signal.sosfiltfilt(sos, x, axis=axis)


def butter_lowpass(x, fs_hz, low_hz, order=15, axis=-1):
    nyq_hz = 0.5 * fs_hz
    sos = scipy.signal.butter(order, low_hz / nyq_hz, btype='lowpass', output='sos')
    return scipy.signal.sosfiltfilt(sos, x, axis=axis)


def butter_highpass(x, fs_hz, high_hz, order=15, axis=-1):
    nyq_hz = 0.5 * fs_hz
    sos = scipy.signal.butter(order, high_hz / nyq_hz, btype='highpass', output='sos')
    return scipy.signal.sosfiltfilt(sos, x, axis=axis)


def gauss_smooth(x, sigma):

    n = len(x)
    n_nextpow2 = int(2 ** np.ceil(np.log2(n)))

    f = 2 * np.pi * np.fft.fftfreq(n_nextpow2)
    ft = np.exp(-0.5 * (f * sigma) ** 2)
    x_smooth = np.fft.ifft(ft * np.fft.fft(x, n_nextpow2))[:n]

    if np.isrealobj(x):
        return x_smooth.real
    else:
        return x_smooth


@njit
def find_extrema(x):
    """
    Finds peaks and troughs in x and stores them in m. Ensures that no two peaks exist without a trough between them
    and visa-versa. Also ensures that find_extrema(m_pos, x) and find_extrema(m_neg, -x) implies that m_pos == -m_neg.
    :param x: input array for which to inspect for peaks and troughs, must of length >= 2.
    :return: int8 array equal length to x, storing values -1 for trough, 1 for peak, and 0 for neither
    """

    # although the word "gradient" or "grad" is used here,
    # it is meant as the gradient sign or direction only rather than the sign and magnitude of the gradient

    m = np.empty(len(x), dtype=np.int8)

    # if negative gradient, then we consider the end a peak, positive gradient is a trough, otherwise neither
    m[0] = int(np.sign(x[0] - x[1]))

    grad_mem = 0
    for i in range(1, len(x) - 1):

        # obtain the direction of the gradient before and after i
        grad_prev = int(np.sign(x[i] - x[i - 1]))
        grad_next = int(np.sign(x[i + 1] - x[i]))

        # carry the last non-zero gradient through if we're in a plateau (unless grad_mem is also 0 from start)
        if grad_prev == 0:
            grad_prev = grad_mem

        # p = grad_prev (could be current or carried over from grad_mem)
        # n = grad_next
        # a = any (can be either -1, 0, or 1, inconsequential which)
        #
        #          can get this            by using this
        #   p   n  ->   m        n-p   p*n   (n-p)*p*n
        #  ----------------------------------------------
        #   0   a  ->   0         a     0        0
        #   a   0  ->   0        -a     0        0
        #  -1  -1  ->   0         0     1        0
        #   1   1  ->   0         0     1        0
        #  -1   1  ->  -1         2    -1       -2
        #   1  -1  ->   1        -2    -1        2

        # m[i] will contain 1 for a peak, -1 for a trough, and 0 if neither, based on the above table
        m[i] = np.sign((grad_next - grad_prev) * grad_prev * grad_next)

        # remember the gradient so that it may be carried forward when we enter a plateau
        if grad_prev != 0:
            grad_mem = grad_prev

    # if positive gradient, then we consider the end a peak, negative gradient is a trough, otherwise neither
    m[-1] = int(np.sign(x[-1] - x[-2]))

    return m


def mean_coh_logit(coh, weights=None, axis=None):

    # logit transform of R, ensuring to nan out any infinities
    z = logit(np.sqrt(coh))
    z[np.isinf(z)] = np.nan

    if axis is None:
        z = np.nanmean(z)
    else:
        # this is needed since nanmean doesn't accept a tuple as the axis argument, so we need to loop over each axis
        if not isinstance(axis, collections.Iterable):
            axis = (axis, )

        # perform the mean over each desired axis
        zm = np.ma.array(z, mask=np.isnan(z))
        zm = np.ma.average(zm, axis=axis, weights=weights)
        z = zm.filled()

    # inverse logit transform, returning to R^2
    return expit(z) ** 2


def mean_coh_fisher(coh, weights=None, axis=None):

    # Fisher transform, ensuring to nan out any infinities
    z = np.arctanh(np.sqrt(coh))
    z[np.isinf(z)] = np.nan

    if axis is None:
        z = np.nanmean(z)
    else:
        # this is needed since nanmean doesn't accept a tuple as the axis argument, so we need to loop over each axis
        if not isinstance(axis, collections.Iterable):
            axis = (axis, )

        # perform the mean over each desired axis
        zm = np.ma.array(z, mask=np.isnan(z))
        zm = np.ma.average(zm, axis=axis, weights=weights)
        z = zm.filled()

    # inverse Fisher transform
    return np.tanh(z) ** 2


def noise_spec_func(t, spec_func):

    dt = t[1] - t[0]
    f = np.fft.fftfreq(2 * len(t) + 1, d=dt)

    phases = np.random.uniform(0, 2 * np.pi, len(f))
    phases[:len(t)] = phases[len(t)+1:]
    c = spec_func(f) * np.exp(1j * phases)

    c[0] = 0
    c[-1] = 0

    y = np.fft.ifft(c)[:len(t)].real

    return y, f[:len(t)]


def decim_half(x, is_time=False, reduce='mean'):
    # TODO currently runs on axis=0, allow an "axis" keyword to run on other axes
    if x.shape[0] % 2 != 0:
        x = x[:-1, ...]
    if is_time:
        x = x[::2, ...]
    else:
        if reduce == 'mean':
            x = 0.5 * (x[::2, ...] + x[1::2, ...])
        elif reduce == 'max':
            x = np.maximum(x[::2, ...], x[1::2, ...])
        elif reduce == 'min':
            x = np.minimum(x[::2, ...], x[1::2, ...])
    return x


def scale01(x):
    return (x - x.min()) / (x.max() - x.min())


@njit
def viterbi(start_lp, trans_lp, emit_lp):
    """
    Viterbi algorithm for n samples and k states
    :param start_lp: size k array of initial log probabilities (for the state just prior to the first emit sample),
    :param trans_lp: size k*k array of transition log probabilities with trans_lp[b, a] from state a to b,
    :param emit_lp: size n*k array of emission log probabilities,
    :return: pair (seq, lp) where seq is a size n array of most-likely hidden states in {0, 1, ..., k-1}, and lp is the
    log probability of the sequence.
    """

    n, k = emit_lp.shape

    lp = np.zeros_like(emit_lp)

    # state_from[i, a] is the state at i-1 that we transitioned from to get to state a
    states_from = np.zeros((n, k), dtype=np.int64)

    # forward (initial step)
    for state_to in range(k):
        lp_s = emit_lp[0] + start_lp
        state_from = np.argmax(lp_s)
        states_from[0, state_to] = state_from
        lp[0, state_to] = lp_s[state_from]

    # forward (remaining steps)
    for i in range(1, n):
        for state_to in range(k):
            lp_s = emit_lp[i] + lp[i - 1] + trans_lp[state_to]
            state_from = np.argmax(lp_s)
            states_from[i, state_to] = state_from
            lp[i, state_to] = lp_s[state_from]

    # backward
    seq = np.zeros(n, dtype=np.int64)
    i = n - 1
    seq[i] = np.argmax(lp[i])
    while i >= 0:
        seq[i - 1] = states_from[i, seq[i]]
        i -= 1

    return seq, np.max(lp[-1])


def xcorr(x, y, normalized=True):
    """
    Cross-correlation between x and y using numpy.correlate(x, y, mode='full'). This results in lags where a negative
    lag means x comes before y, and positive lag x comes after y. As a mneumonic, think of it as a subtraction
    t_x - t_y, with a lower time for x meaning it comes before y and has a negative lag.
    :param x: size n signal array,
    :param y: size n signal array,
    :param normalized: bool on whether to use the normalized cross-correlation (default True),
    :return: pair (xc, lags) where xc is an array of length 2*n-1 of the values of correlation, and lags is an array of
    length 2*n-1 of lags in index units, with lags[n-1] representing 0 lag.
    """

    # correlate y and x,
    #   a negative lag means x occurs before y
    #   a positive lag means x occurs after y
    xc = np.correlate(x, y, mode='full')

    n = len(x)
    lag0_idx = n - 1

    if normalized:
        x_auto = np.correlate(x, x, mode='full')
        y_auto = np.correlate(y, y, mode='full')
        xc /= np.sqrt(x_auto[lag0_idx] * y_auto[lag0_idx])

    lags = (np.arange(2 * n - 1) - lag0_idx)

    return xc, lags


def consecutive(data, stepsize=1):
    """
    Thanks to https://stackoverflow.com/a/7353335/142712.
    Find consecutive sequences in data. To get indexes of subsequences satsfying some predicate (e.g. > 0) do
    for example: idxs = consecutive(np.where(x > 0)[0])
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
