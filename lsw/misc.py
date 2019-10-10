"""
Miscellaneous functions that aren't generalisable, but I often need them.
"""

import pandas as pd


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

    return times, marks, pres
