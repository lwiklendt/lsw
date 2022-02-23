import datetime
import logging
import os
from pathlib2 import Path
import pickle
import pprint
import shutil
import sys
from typing import Union

import matplotlib.pyplot as plt
from matplotlib import gridspec
from cmdstanpy import CmdStanModel
import numpy as np

from lsw.file import ensure_path


def sample(src_stan_file: Union[str, Path], data: dict,
           output_dir: Union[str, Path] = None,
           sample_kwargs: dict = None, compile_kwargs: dict = None,
           force_resample = False,
           **other_sample_kwargs):
    """
    MCMC samples from a CmdStanModel.

    :param src_stan_file: Stan source code file. Will be compied over to output dir
    :param data: dict mapping from data block variable names to data
    :param output_dir: Location to where artefacts will be written. Uses Path(src_stan_file).parent if None
    :param sample_kwargs: args that can modify the output, such as seed, adapt_delta etc.
        These kwargs will be included in the hash to check whether we can use the cached
        samples, or whether we need to resample the model. For other args that do not
        change the posterior such as `refresh`, supply as a kwarg via **other_stan_kwargs
    :param compile_kwargs: kwargs passed to CmdStanModel.
        E.g. compile_kwargs = {'cpp_options': {'STAN_THREADS': True, 'STAN_OPENCL': True}}
    :param force_resample: force a resample even if otherwise it would not be needed
    :param other_sample_kwargs: kwargs to CmdStanModel.sample() that do not change the posterior
        distribution, such as refresh or show_progress. Any changes to these parameters will
        not trigger a resampling.
    :return: dict of variable names mapping to posterior samples.
    """

    # TODO
    #  - Show which parts of the model source code changed?

    # Create filenames and dirnames.
    src_stan_file = Path(src_stan_file)
    model_name = src_stan_file.stem
    output_dir = src_stan_file.parent / model_name if output_dir is None else Path(output_dir)
    dst_stan_file = output_dir / src_stan_file.name
    samples_file = output_dir / 'samples.pkl'
    data_file = output_dir / 'data.pkl'
    compile_kwargs_file = output_dir / 'compile_kwargs.pkl'
    sample_kwargs_file = output_dir / 'sample_kwargs.pkl'
    log_file = output_dir / 'log.txt'
    traces_plot_file = output_dir / 'traces.png'
    ensure_path(output_dir)

    # Sanitise kwargs.
    sample_kwargs = {} if sample_kwargs is None else sample_kwargs
    compile_kwargs = {} if compile_kwargs is None else compile_kwargs

    # ======================= #
    # ----- Compilation ----- #
    # ======================= #

    # If compile_kwargs is the same as cached, then we don't force compile.
    compile = 'force'
    if compile_kwargs_file.exists():
        with open(compile_kwargs_file, 'rb') as f:
            compile_kwargs_cached = pickle.load(f)
        if compile_kwargs_cached == compile_kwargs:
            # Turn off forced compile, and go back to weaker compile determination based only on source code change.
            compile = True

    if compile == 'force':
        print('cached compile_kwargs missing or different')

    # Setup logging, DEBUG to file, INFO to stdout.
    logger = logging.getLogger('cmdstanpy')
    logger.setLevel(logging.DEBUG)
    if len(logger.handlers) == 0:
        fh = logging.FileHandler(filename=log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)

    # Copy stan file over to output directory.
    shutil.copy2(src_stan_file, dst_stan_file)

    # Create model, possibly recompiling.
    pre_compile_time = datetime.datetime.now().timestamp()
    m = CmdStanModel(stan_file=dst_stan_file, compile=compile, **compile_kwargs)
    exe_file = Path(m.exe_file)
    new_exe = False
    if exe_file.exists():
        if os.path.getmtime(exe_file) > pre_compile_time:
            new_exe = True

    # Write compile_kwargs.
    with open(compile_kwargs_file, 'wb') as f:
        pickle.dump(compile_kwargs, f, protocol=4)

    # Only want debug level for the compilation
    logger.setLevel(logging.INFO)

    # ==================== #
    # ----- Sampling ----- #
    # ==================== #

    # Identify reason for resampling for user feedback.
    resampling_reasons = []
    if force_resample:
        resampling_reasons.append('force resample')
    if new_exe:
        resampling_reasons.append('model recompiled')
    if not samples_file.exists():
        resampling_reasons.append('samples missing')
    if not sample_kwargs_file.exists():
        resampling_reasons.append('sample_kwargs cache missing')
    if not data_file.exists():
        resampling_reasons.append('data cache missing')

    # If sample_kwargs or data have not changed, then load and return the cached samples.
    if len(resampling_reasons) == 0:

        # Check to see if the given sample_kwargs is the same as the cached sample_kwargs.
        with open(sample_kwargs_file, 'rb') as f:
            sample_kwargs_same = (sample_kwargs == pickle.load(f))

        # Check to see if the given data is the same as the cached data.
        with open(data_file, 'rb') as f:
            data_cache = pickle.load(f)
            try:
                np.testing.assert_equal(data, data_cache)
                data_same = True
            except AssertionError as e:
                data_same = False

        # If all checks passed then we can load the cached samples,
        if sample_kwargs_same and data_same:
            with open(samples_file, 'rb') as f:
                return pickle.load(f)

        # otherwise, append the reason for failed checks.
        else:
            if not sample_kwargs_same:
                resampling_reasons.append('sample_kwargs changed')
            if not data_same:
                resampling_reasons.append('data changed')

    # Report reasons for resampling.
    print(f'Resampling due to: {", ".join(resampling_reasons)}')

    # Perform sampling.
    sampling_start = datetime.datetime.now()
    result = m.sample(data=data, **{**sample_kwargs, **other_sample_kwargs})
    sampling_end = datetime.datetime.now()
    print(f'Sampling complete in: {sampling_end - sampling_start}')

    # Write samples.
    print(f'Pickling posterior...', end='')
    start = datetime.datetime.now()
    samples = result.stan_variables()
    with open(samples_file, 'wb') as f:
        pickle.dump(samples, f, protocol=4)
    print(f'done ({datetime.datetime.now() - start})')

    # Write sample_kwargs.
    with open(sample_kwargs_file, 'wb') as f:
        pickle.dump(sample_kwargs, f, protocol=4)

    # Write data cache.
    with open(data_file, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    # ================================= #
    # ----- Diagnosis and summary ----- #
    # ================================= #

    # Overview and args.
    summary = f'sampling started: {sampling_start}\n' \
              f'sampling ended  : {sampling_end}\n' \
              f'sampling elapsed: {sampling_end - sampling_start}\n\n' \
              f'stan_kwargs={pprint.pformat(sample_kwargs)}\n\n'

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

    with open(output_dir / 'summary.txt', 'w') as f:
        f.write(summary)

    # ======================= #
    # ----- Plot traces ----- #
    # ======================= #

    fig = plot_traces(samples, df_summary)
    fig.savefig(traces_plot_file)
    plt.close(fig)

    print(f'output_dir = {output_dir}')

    return samples


def plot_traces(samples, df_summary, params=None):

    usetex = plt.rcParams['text.usetex']
    plt.rcParams['text.usetex'] = False

    # if no parameters supplied, get the list from the samples
    if params is None:
        params = list(samples.keys())

    # plot params that are small enough to not overwhelm matplotlib using arbitrary threshold of prod(shape) < 1e6
    params = [p for p in params if np.prod(samples[p].shape) < 1e6]

    fig = plt.figure(figsize=(12, 2 * len(params) + 3))
    gs = gridspec.GridSpec(len(params) + 2, 2, width_ratios=[5, 1])
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
