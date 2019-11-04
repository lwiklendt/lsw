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


def noise_inv_power(t, gamma=1.0, min_freq=0.0, max_freq=np.inf):

    def spec_func(f_):
        f_ = np.abs(f_)
        f_[f_ > 0] **= -gamma
        f_[f_ <= min_freq] = 0
        f_[f_ > max_freq] = 0
        return f_

    y, f = noise_spec_func(t, spec_func)

    # normalise to unit standard deviation
    y = y / np.std(y)

    return y, f


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
