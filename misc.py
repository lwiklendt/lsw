"""
Miscellaneous functions that aren't generalisable, but I often need them.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import xlrd

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
        p[chan, :] -= signal.baseline_gauss(p[chan, :], sigma_samples, iters)
    thread.parexec(exec_func, p.shape[0])

    # synchronous activity removal
    if sync_rem:
        p = np.maximum(0, p - np.maximum(0, np.median(p, axis=0, keepdims=True)))

    return p


def load_plothrm_detailed(filename):
    """
    Load an xlsx file that contains sheets that were pasted from "Export Detailed" in PlotHRM.
    """

    filename = Path(filename)

    book = xlrd.open_workbook(filename)
    xlsx_file = pd.ExcelFile(filename)

    df_seq = None
    df_meta = None

    for sheetidx, sheetname in enumerate(xlsx_file.sheet_names):

        print(f'Parsing "{filename}" sheet "{sheetname}"', flush=True)

        try:

            sheetnumber = sheetidx + 1
            worksheet = book.sheet_by_name(sheetname)

            # try to load a value from the row and column
            def value_default(row, col, d):
                if row >= worksheet.nrows or col >= worksheet.ncols:
                    return d
                s = worksheet.cell_value(row, col)
                if type(s) is str and len(s) == 0:
                    return d
                else:
                    return s

            # load metadata
            meta = dict()
            meta['sheetname'] = sheetname
            meta['sheetnumber'] = sheetnumber
            meta['plothrm_version'] = str(worksheet.cell_value(0, 1))
            meta['data_filename']   = str(worksheet.cell_value(1, 1))
            meta['seq_filename']    = str(worksheet.cell_value(2, 1))
            meta['nchan']           = int(worksheet.cell_value(3, 1))
            meta['nsamp']           = int(worksheet.cell_value(4, 1))
            meta['sampling_hz']     = float(worksheet.cell_value(5, 1))
            meta['opt_zero_above']       = float(value_default(6,  2, np.nan))
            meta['opt_remove_baselines'] = bool (value_default(6,  4, 0))
            meta['opt_remove_sync']      = bool (value_default(6,  6, 0))
            meta['opt_channel_smooth']   = bool (value_default(6,  8, 0))
            meta['opt_sample_smooth']    = float(value_default(6, 10, np.nan))
            meta['height_res']  = float(worksheet.cell_value(7, 1))
            meta['sync_bound']  = float(worksheet.cell_value(8, 1))
            # TODO load region data at rows 9, 10, 11 (Excel 1-based index rows 10-12)

            for k, v in meta.items():
                meta[k] = [v]
            meta = pd.DataFrame(meta)

            if df_meta is None:
                df_meta = meta
            else:
                df_meta = df_meta.append(meta)

            # load sequence data
            df = pd.read_excel(filename, sheetname, header=13, na_values=['ERROR', 'Infinity'])
            df['sheetname'] = sheetname
            df['sheetnumber'] = sheetnumber
            if df_seq is None:
                df_seq = df
            else:
                df_seq = df_seq.append(df)

        except Exception as e:
            print(f'  error parsing worksheet "{sheetname}": {e}')
            continue

    return df_seq.reset_index(drop=True), df_meta.reset_index(drop=True)
