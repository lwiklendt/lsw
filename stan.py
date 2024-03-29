import datetime
import logging
import os
from pathlib import Path
import pickle
import pprint
import shutil
import sys
import tempfile
from typing import Union

import matplotlib.pyplot as plt
from matplotlib import gridspec
from cmdstanpy import CmdStanModel
import numpy as np

from lsw.file import ensure_path


def negbinom_mu_phi_to_alpha_beta(mu, phi):
    """
    Converts from Stan's alternative paramaterisation (mu, phi) to the (alpha, beta) parameterisation.
    Where mean: mu = alpha/beta, variance: mu + mu**2/phi = alpha * (beta + 1) / beta**2.
    :param mu: parameter 1
    :param phi: parameter 2
    :return: tuple (alpha, beta)
    """
    # Check:
    #   alpha = phi
    #   beta = phi/mu
    #   mean:
    #     mu = alpha / beta
    #        = phi / (phi / mu)
    #        = mu
    #   variance:
    #     mu + mu**2/phi = alpha * (beta + 1) / beta**2
    #                    = phi * (phi/mu + 1) / (phi/mu)**2
    #                    = phi * (phi/mu + 1) * mu**2/phi**2
    #                    = (phi**2/mu + phi) * (mu**2/phi**2)
    #                    = (phi**2/mu)*(mu**2/phi**2) + phi*(mu**2/phi**2)
    #                    = mu + mu**2/phi
    alpha = phi
    beta = phi / mu
    return alpha, beta


def negbinom_mu_phi_to_numpy(mu, phi):
    """
    Converts from Stan's alternative paramaterisation (mu, phi) to the (n, p) parameterisation in
    numpy and scipy, for n successes and probability p of success.
    Where mean = mu, variance = mu + mu**2/phi.
    :param mu: parameter 1
    :param phi: parameter 2
    :return: tuple (n, p)
    """
    # Check:
    #   variance = mu + mu**2/phi
    #   n = mu**2 / (var - mu)
    #     = mu**2 / (mu + mu**2/phi - mu)
    #     = mu**2 / (mu**2/phi)
    #     = phi
    #   p = n / (n + mu)
    #     = phi / (phi + mu)
    n = phi
    p = phi / (phi + mu)
    return n, p


def sample(src_stan_filename: Union[str, Path],
           data: dict = None,
           output_dirname: Union[str, Path] = None,
           sample_kwargs: dict = None,
           compile_kwargs: dict = None,
           force_resample = False,
           method = 'mcmc',
           **other_sample_kwargs):
    """
    MCMC samples from a CmdStanModel.

    :param src_stan_filename: Stan source code filename. Will be copied over to output dir
    :param data: dict mapping from data block variable names to data. If None then attempt to obtain cached samples.
    :param output_dirname: Location to where artefacts will be written. Uses Path(src_stan_file).parent if None
    :param sample_kwargs: args that can modify the output, such as seed, adapt_delta etc.
        These kwargs will be included in the hash to check whether we can use the cached
        samples, or whether we need to resample the model. For other args that do not
        change the posterior such as `refresh`, supply as a kwarg via **other_stan_kwargs
    :param compile_kwargs: kwargs passed to CmdStanModel.
        E.g. compile_kwargs = {'cpp_options': {'STAN_THREADS': True, 'STAN_OPENCL': True}}
    :param force_resample: force a resample even if otherwise it would not be needed
    :param method: inference method, one of ['mcmc', 'vb'].
    :param other_sample_kwargs: kwargs to CmdStanModel.sample() that do not change the posterior
        distribution, such as refresh or show_progress. Any changes to these parameters will
        not trigger a resampling.
    :return: dict of variable names mapping to posterior samples.
    """

    # Create filenames and dirnames.
    src_stan_filename = Path(src_stan_filename)
    model_name = src_stan_filename.stem
    output_dirname = src_stan_filename.parent / model_name if output_dirname is None else Path(output_dirname)
    dst_stan_filename = output_dirname / src_stan_filename.name
    samples_filename = output_dirname / 'samples.pkl'
    data_file = output_dirname / 'data.pkl'
    compile_kwargs_filename = output_dirname / 'kwargs_compile.pkl'
    sample_kwargs_filename = output_dirname / 'kwargs_sample.pkl'
    compile_log_filename = output_dirname / 'log_compile.txt'
    sample_log_filename = output_dirname / 'log_sample.txt'
    traces_plot_filename = output_dirname / 'traces.png'
    ensure_path(output_dirname)

    # Ensure kwargs are dict in place of None.
    sample_kwargs = {} if sample_kwargs is None else sample_kwargs
    compile_kwargs = {} if compile_kwargs is None else compile_kwargs

    # ======================= #
    # ----- Compilation ----- #
    # ======================= #

    # If compile_kwargs is the same as cached, then we don't force compile.
    compile = 'force'
    if compile_kwargs_filename.exists():
        with open(compile_kwargs_filename, 'rb') as f:
            compile_kwargs_cached = pickle.load(f)
        if compile_kwargs_cached == compile_kwargs:
            # Turn off forced compile, and go back to weaker compile determination based only on source code change.
            compile = True
    if compile == 'force':
        print('cached compile_kwargs missing or different')

    # Setup logging, DEBUG by default and INFO to stdout.
    logger = logging.getLogger('cmdstanpy')
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    # TODO
    #  - Diff model source code to show what has changed before copying over?

    # Copy stan file over to output directory.
    shutil.copy2(src_stan_filename, dst_stan_filename)

    # Prepare and add compile log file handler.
    # We don't know whether CmdStanModel will need to compile a new file or not, so we store the log
    # in a temporary file and if a recompilation was indeed performed we copy over that file to
    # compile_log_file.
    with tempfile.TemporaryDirectory() as tempdir:
        compile_log_tempfilename = Path(tempdir)/'log_compile.txt'
        compile_log_handler = logging.FileHandler(filename=compile_log_tempfilename)
        compile_log_handler.setLevel(logging.DEBUG)
        logger.addHandler(compile_log_handler)

        # Create model, possibly recompiling.
        pre_compile_time = datetime.datetime.now().timestamp()
        m = CmdStanModel(stan_file=dst_stan_filename, compile=compile, **compile_kwargs)
        exe_file = Path(m.exe_file)
        new_exe = False
        if exe_file.exists():
            if os.path.getmtime(exe_file) > pre_compile_time:
                new_exe = True

        # Compilation complete so remove and close the compile log handler.
        logger.removeHandler(compile_log_handler)
        if new_exe:
            # If there was a compilation, store the log file
            shutil.copy2(compile_log_tempfilename, compile_log_filename)
        compile_log_handler.close()

    # Write compile_kwargs.
    with open(compile_kwargs_filename, 'wb') as f:
        pickle.dump(compile_kwargs, f, protocol=4)

    # ==================== #
    # ----- Sampling ----- #
    # ==================== #

    # Identify reason for resampling.
    resampling_reasons = []
    if force_resample:
        resampling_reasons.append('force resample')
    if new_exe:
        resampling_reasons.append('model recompiled')
    if not samples_filename.exists():
        resampling_reasons.append('samples cache missing')
    if not sample_kwargs_filename.exists():
        resampling_reasons.append('sample_kwargs cache missing')
    # Data cache is allowed to be missing if no data is supplied, since user just wants to return samples rather than
    # check whether resampling is required based on change in data.
    if not data_file.exists() and data is not None:
        resampling_reasons.append('data cache missing')

    # If there is no reason to resample yet then try loading the cached samples.
    if len(resampling_reasons) == 0:

        # Check to see if the given sample_kwargs is the same as the cached sample_kwargs.
        with open(sample_kwargs_filename, 'rb') as f:
            sample_kwargs_same = (sample_kwargs == pickle.load(f))

        # Check to see if the given data is the same as the cached data.
        # If data is not supplied assume data is the same as the cached samples.
        data_same = data is None
        if data_file.exists():
            with open(data_file, 'rb') as f:
                data_cache = pickle.load(f)
                try:
                    np.testing.assert_equal(data, data_cache)
                    data_same = True
                except AssertionError:
                    data_same = False

        # If all checks passed then we can load the cached samples.
        if sample_kwargs_same and data_same:
            with open(samples_filename, 'rb') as f:
                print('Returning cached samples')
                return pickle.load(f)

        # otherwise, append the reason for failed checks.
        else:
            if not sample_kwargs_same:
                resampling_reasons.append('sample_kwargs changed')

    # Report reasons for resampling.
    print(f'Resampling due to: {", ".join(resampling_reasons)}')

    # If we have reached this point it means some kwargs changed or there are no samples, either
    # way we need data to resample.
    if data is None:
        raise RuntimeError('No data supplied for sampling')

    # We need to recompile, so prepare the sample log file.
    sample_log_handler = logging.FileHandler(filename=sample_log_filename, mode='w')
    sample_log_handler.setLevel(logging.DEBUG)
    logger.addHandler(sample_log_handler)

    # Perform sampling.
    sampling_start = datetime.datetime.now()
    if method == 'mcmc':
        result = m.sample(data=data, **{**sample_kwargs, **other_sample_kwargs})
    elif method == 'vb':
        result = m.variational(data=data, **{**sample_kwargs, **other_sample_kwargs})
    else:
        raise RuntimeError(f'unknown inference method: {method}')
    sampling_end = datetime.datetime.now()
    print(f'Sampling complete in: {sampling_end - sampling_start}')

    # Sampling complete so remove and close the sample log handler.
    logger.removeHandler(sample_log_handler)
    sample_log_handler.close()

    # Write samples.
    print(f'Pickling posterior...', end='')
    start = datetime.datetime.now()
    if method == 'mcmc':
        samples = result.stan_variables()
    elif method == 'vb':
        samples = cmdstanvb_extract(result)
    else:
        raise RuntimeError(f'unknown inference method: {method}')
    with open(samples_filename, 'wb') as f:
        pickle.dump(samples, f, protocol=4)
    print(f'done ({datetime.datetime.now() - start})')

    # Writing sample_kwargs and data cache is done only after writing the samples cache file so that we know that the
    # caches and supplied values match.

    # Write sample_kwargs.
    with open(sample_kwargs_filename, 'wb') as f:
        pickle.dump(sample_kwargs, f, protocol=4)

    # Write data cache.
    if data is not None:
        with open(data_file, 'wb') as f:
            pickle.dump(data, f, protocol=4)

    # ================================= #
    # ----- Diagnosis and summary ----- #
    # ================================= #

    # TODO report to stdout the number of divergences and any other diagnostic problems.
    #  - Even better would be to structured diagnostic output in something like json format to a file,
    #    but only showing problems such as rhats larger than certain values.
    #  - We would like to ask something like has_problems() and get a bool.

    # Overview and args.
    summary = f'sampling started: {sampling_start}\n' \
              f'sampling ended  : {sampling_end}\n' \
              f'sampling elapsed: {sampling_end - sampling_start}\n\n' \
              f'stan_kwargs={pprint.pformat(sample_kwargs)}\n\n'

    if method == 'mcmc':
        # Diagnosis.
        print(f'Diagnosing...', end='')
        start = datetime.datetime.now()
        summary += f'{result.diagnose()}\n\n'
        print(f'done ({datetime.datetime.now() - start})')

        # Variable summary.
        print(f'Summarising variables...', end='')
        start = datetime.datetime.now()
        df_summary = result.summary()
        df_summary_non_nan = df_summary.dropna()
        df_summary_nan = df_summary[df_summary.isnull().any(1)]
        summary += f'Non-NaN summary:\n{df_summary_non_nan.to_string()}\n\n'
        summary += f'NaN summary:\n{df_summary_nan.to_string()}'
        print(f'done ({datetime.datetime.now() - start})')
    elif method == 'vb':
        df_summary = None
    else:
        raise RuntimeError(f'unknown inference method: {method}')

    with open(output_dirname / 'summary.txt', 'w') as f:
        f.write(summary)

    # ======================= #
    # ----- Plot traces ----- #
    # ======================= #

    fig = plot_traces(samples, df_summary)
    fig.savefig(traces_plot_filename)
    plt.close(fig)

    print(f'output_dir = {output_dirname}')

    return samples


def plot_traces(samples, df_summary=None, params=None):

    usetex = plt.rcParams['text.usetex']
    plt.rcParams['text.usetex'] = False

    # if no parameters supplied, get the list from the samples
    if params is None:
        params = list(samples.keys())

    # plot params that are small enough to not overwhelm matplotlib using arbitrary threshold of prod(shape) < 1e6
    params = [p for p in params if np.prod(samples[p].shape) < 1e6]

    fig = plt.figure(figsize=(12, 2 * len(params) + 1 + 2 * (df_summary is not None)))
    gs = gridspec.GridSpec(len(params) + 2 * (df_summary is not None), 2, width_ratios=[5, 1])
    for i, k in enumerate(params):
        print(f'plotting {k}... ', end='', flush=True)
        s = samples[k]
        shp = s.shape
        is_multi = len(s.shape) > 1
        if is_multi:
            s = np.reshape(s, (s.shape[0], -1))

        # plot traces
        ax = fig.add_subplot(gs[i, 0])
        if len(shp) > 1:
            ax.set_ylabel(f'{k}\n{shp[1:]}')
        else:
            ax.set_ylabel(f'{k}')
        ax.plot(s, lw=1, alpha=0.7 if is_multi else 0.9)

        # plot histograms
        ax = fig.add_subplot(gs[i, 1])
        if is_multi:
            for j in range(s.shape[1]):
                ax.hist(s[:, j], bins=50, alpha=0.8)
        else:
            ax.hist(s, bins=50)

        print('done', flush=True)

    if df_summary is not None:
        # plot N_Eff
        print(f'plotting n_eff', flush=True)
        ax = fig.add_subplot(gs[-2, :])
        n_eff = df_summary['N_Eff'].values
        ax.hist(n_eff[np.isfinite(n_eff)], bins=100)
        ax.set_xlabel('Number of effective samples')
        ax.set_ylabel('Param Count')

        # plot Rhat
        print(f'plotting Rhat', flush=True)
        ax = fig.add_subplot(gs[-1, :])
        rhat = df_summary['R_hat'].values
        ax.hist(rhat[np.isfinite(rhat)], bins=100)
        ax.set_xlabel('Rhat')
        ax.set_ylabel('Param Count')

    fig.tight_layout()

    plt.rcParams['text.usetex'] = usetex

    return fig


def cmdstanvb_extract(vb):

    samples = vb.variational_sample

    # Cmdstanpy documentation says it returns a np.ndarray, but it actually returns a pandas DataFrame.
    if type(samples) is not np.ndarray:
        samples = samples.values

    n = samples.shape[0]

    # First pass, calculate the shape of each variable.
    param_shapes = dict()
    for column_name in vb.column_names:
        splt = column_name.split('[')
        name = splt[0]
        if len(splt) > 1:
            # No +1 for shape calculation because cmdstanpy already returns 1-based indexes for vb!
            idxs = [int(i) for i in splt[1][:-1].split(',')]
        else:
            idxs = ()
        param_shapes[name] = np.maximum(idxs, param_shapes.get(name, idxs))

    # Create arrays.
    params = {name: np.nan * np.empty((n, ) + tuple(shape)) for name, shape in param_shapes.items()}

    # Second pass, fill arrays.
    for j, column_name in enumerate(vb.column_names):
        splt = column_name.split('[')
        name = splt[0]
        if len(splt) > 1:
            idxs = [int(i) - 1 for i in splt[1][:-1].split(',')]  # -1 because cmdstanpy returns 1-based indexes for vb!
        else:
            idxs = ()
        params[name][(..., ) + tuple(idxs)] = samples[:, j]

    return params
