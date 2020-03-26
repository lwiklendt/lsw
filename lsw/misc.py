"""
Miscellaneous functions that aren't generalisable, but I often need them.
"""

import pandas as pd
import numpy as np

import lsw.signal
import lsw.thread


def load_hrm_txt(filename):
    """
    Load a High-Resolution Manometry text file saved by John's catheter recording software.
    :param filename: full-path filename of the text file saved with the "<TIME>\t<MARK>\t<P0>\t<P1>\t...\t<PM>" format
    :return: t, c, p: numpy arrays of n time-samples, with time t:(n,), marks c:(n,), pressures p:(m,n)
    """
    df = pd.read_csv(filename, header=None, sep='\t')
    x = df.values
    times = x[:, 0]
    marks = x[:, 1]
    pres = x[:, 2:].T

    # sometimes there is an extra column of NANs at the end, so remove it
    if np.sum(np.isnan(pres[-1])) == len(times):
        pres = pres[:-1, :]

    return times, marks, pres


def clean_pressures(p, sigma_samples, iters, sync_rem):
    """
    Performs baseline and synchronous anomaly removal.
    @param p: (nchan, nsamp) shaped array of pressures
    @param sigma_samples: parameter for lsw.signal.baseline_gauss
    @param iters: parameter for lsw.signal.baseline_gauss
    @param sync_rem: whether to perform synchronous anomaly removal
    @return: cleaned p
    """

    # baseline removal
    def exec_func(chan):
        p[chan, :] -= lsw.signal.baseline_gauss(p[chan, :], sigma_samples, iters)
    lsw.thread.parexec(exec_func, p.shape[0])

    # synchronous activity removal
    if sync_rem:
        p = np.maximum(0, p - np.maximum(0, np.median(p, axis=0, keepdims=True)))

    return p
