import os

from collections import OrderedDict
import datetime
import difflib
from hashlib import md5
import pickle

import matplotlib.pyplot as plt
from matplotlib import gridspec

import pystan
import numpy as np


def check_div(fit):
    """Check transitions that ended with a divergence"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    divergent = [x for y in sampler_params for x in y['divergent__']]
    n = int(sum(divergent))
    N = len(divergent)
    ret = f'{n} of {N} iterations ended with a divergence ({100*n/N}%)'
    if n > 0:
        ret += '\nTry running with larger adapt_delta to remove the divergences'
    return ret


def check_treedepth(fit):
    """Check transitions that ended prematurely due to maximum tree depth limit"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    depths = [x for y in sampler_params for x in y['treedepth__']]
    max_depth = int(np.max(depths))
    # n = sum(1 for x in depths if x == max_depth)
    # N = len(depths)
    # ret = f'top tree depth of {max_depth} accounted for {n} of {N} iterations ({100*n/N}%)'
    ret = f'top tree depth of {max_depth}'
    return ret


def check_energy(fit):
    """Checks the energy Bayesian fraction of missing information (E-BFMI)"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    ret = []
    for chain_num, s in enumerate(sampler_params):
        energies = s['energy__']
        numer = sum((energies[i] - energies[i - 1])**2 for i in range(1, len(energies))) / len(energies)
        denom = np.var(energies)
        if numer / denom < 0.2:
            ret.append(f'Chain {chain_num}: E-BFMI = {numer / denom}'
                       '\nE-BFMI below 0.2 indicates you may need to reparameterize your model')

    return '\n'.join(ret)


def _by_chain(unpermuted_extraction):
    num_chains = len(unpermuted_extraction[0])
    result = [[] for _ in range(num_chains)]
    for c in range(num_chains):
        for i in range(len(unpermuted_extraction)):
            result[c].append(unpermuted_extraction[i][c])
    return np.array(result)


def _shaped_ordered_params(fit):
    ef = fit.extract(permuted=False, inc_warmup=False)  # flattened, unpermuted, by (iteration, chain)
    ef = _by_chain(ef)
    ef = ef.reshape(-1, len(ef[0][0]))
    ef = ef[:, 0:len(fit.flatnames)]  # drop lp__
    shaped = {}
    idx = 0
    for dim, param_name in zip(fit.par_dims, fit.extract().keys()):
        length = int(np.prod(dim))
        shaped[param_name] = ef[:, idx:idx + length]
        shaped[param_name].reshape(*([-1] + dim))
        idx += length
    return shaped


def partition_div(fit):
    """ Returns parameter arrays separated into divergent and non-divergent transitions"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    div = np.concatenate([x['divergent__'] for x in sampler_params]).astype('int')
    params = _shaped_ordered_params(fit)
    nondiv_params = dict((key, params[key][div == 0]) for key in params)
    div_params = dict((key, params[key][div == 1]) for key in params)
    return nondiv_params, div_params


def compile_model(filename, model_name=None, cache_path=None):
    """This will automatically cache models - great if you're just running a
    script on the command line.

    See http://pystan.readthedocs.io/en/latest/avoiding_recompilation.html"""

    with open(filename) as f_src:
        model_code = f_src.read()
        code_hash = md5(model_code.encode('utf8')).hexdigest()
        if model_name is None:
            model_name = os.path.splitext(os.path.split(filename)[-1])[0]
            print(f'Inferred model name: {model_name}')
        cache_fn = f'{model_name}-{code_hash}.pkl'
        if cache_path is not None:
            cache_fn = os.path.join(cache_path, cache_fn)
        if os.path.exists(cache_fn):
            sm = pickle.load(open(cache_fn, 'rb'))
            print('Using cached StanModel')
        else:
            sm = pystan.StanModel(model_name=model_name, model_code=model_code, obfuscate_model_name=False)
            with open(cache_fn, 'wb') as f_pkl:
                pickle.dump(sm, f_pkl, protocol=2)
        return sm


def sample_in_path(outpath, data, code=None, code_filename=None, params=None, method='mcmc', **stan_kwargs):

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    model_cache_filename   = os.path.join(outpath, 'stan_model.pkl')
    model_code_filename    = os.path.join(outpath, 'model.stan')
    samples_cache_filename = os.path.join(outpath, 'samples.pkl')
    diagnostics_filename   = os.path.join(outpath, 'diagnostics.txt')
    traces_filename        = os.path.join(outpath, 'traces.png')

    # if code was not supplied but a filename was, then read the filename
    if code is None and code_filename is not None:
        with open(code_filename, 'r') as f:
            code = f.read()

    # write out stan source code
    with open(model_code_filename, 'w') as f:
        f.write(code)

    # compile stan code
    stan_model = None
    code_digest = md5(code.encode('utf8')).hexdigest()
    if model_cache_filename is not None and os.path.exists(model_cache_filename):
        with open(model_cache_filename, 'rb') as f:
            cached_digest, cached_stan_code, stan_model = pickle.load(f)
        if cached_digest != code_digest:
            stan_model = None
            print('Stan model code is different to the cached version (- cached, + curr):')
            # print the diff between cached and current code
            result = difflib.unified_diff(cached_stan_code.splitlines(), code.splitlines(), n=0, lineterm='')
            print('\n'.join(list(result)[2:]))  # [2:] is to remove the control lines '---' and '+++'
            print('recompiling...')
    if stan_model is None:
        compile_start = datetime.datetime.now()
        stan_model = pystan.StanModel(model_code=code)
        print(f'Elapsed compilation time: {datetime.datetime.now() - compile_start}')
        if model_cache_filename is not None:
            with open(model_cache_filename, 'wb') as f:
                pickle.dump((code_digest, code, stan_model), f, protocol=2)

    # check if we need to resample, or can load from cache
    needs_sample = True
    samples = None
    dict_digest = lambda d: md5(pickle.dumps(OrderedDict(sorted(d.items())))).hexdigest()
    data_digest = _hash_dict_with_numpy_arrays(data)  # dict_digest doesn't seem to always produce the same hash for the same numpy arrays
    kwargs_digest = dict_digest(stan_kwargs)
    if samples_cache_filename is not None and os.path.exists(samples_cache_filename):
        with open(samples_cache_filename, 'rb') as f:
            cached_data_digest, cached_kwargs_digest, cached_code_digest, samples = pickle.load(f)
        data_changed = cached_data_digest != data_digest
        kwargs_changed = cached_kwargs_digest != kwargs_digest
        code_changed = cached_code_digest != code_digest
        needs_sample = data_changed or kwargs_changed or code_changed
        if needs_sample:
            changes = ['model changed'] if code_changed else []
            changes += ['data changed'] if data_changed else []
            changes += ['stan kwargs changed'] if kwargs_changed else []
            print(f'{", ".join(changes)}: resampling...')

    # sample posterior
    if needs_sample:

        if (method == 'mcmc' or method == 'vb') and stan_kwargs.get('init', None) == 'optimizing':
            init_kwargs = dict(as_vector=True)
            print(f'Initializing with LBFGS')
            if 'seed' in stan_kwargs:
                init_kwargs['seed'] = stan_kwargs['seed']
            init = stan_model.optimizing(data, **init_kwargs)
            init = [init, ] * stan_kwargs.get('chains', 1)
            stan_kwargs['init'] = init

        # perform mcmc sampling
        if method == 'mcmc':
            timer_start = datetime.datetime.now()
            print(f'Started MCMC at {timer_start}')
            stan_fit = stan_model.sampling(data=data, pars=params, **stan_kwargs)
            print(f'Elapsed MCMC {datetime.datetime.now() - timer_start}')

            samples = stan_fit.extract()

            # store diagnostics in samples dict
            if stan_fit is not None:
                samples.seed = stan_fit.get_seed()
                samples.elapsed_time = datetime.datetime.now() - timer_start
                samples.check_fit = str(stan_fit)
                samples.check_treedepth = check_treedepth(stan_fit)
                samples.check_energy = check_energy(stan_fit)
                samples.check_div = check_div(stan_fit)
                samples.sampler_params = stan_fit.get_sampler_params()
                summary = stan_fit.summary()
                samples.n_eff = summary['summary'][:, summary['summary_colnames'].index('n_eff')]
                samples.rhat = summary['summary'][:, summary['summary_colnames'].index('Rhat')]
                print('\n'.join([samples.check_treedepth, samples.check_energy, samples.check_div]))

        # perform vb sampling
        elif method == 'vb':
            results = stan_model.vb(data, pars=params, diagnostic_file=diagnostics_filename, **stan_kwargs)
            samples = pystan_vb_extract(results)

        else:
            stan_kwargs['algorithm'] = stan_kwargs['algorithm'].upper()
            samples = stan_model.optimizing(data, as_vector=True, **stan_kwargs)

            # make compatible with sample, by returning as a single sample
            for k, v in samples.items():
                samples[k] = v[np.newaxis, ...]

        # cache samples
        with open(samples_cache_filename, 'wb') as f:
            pickle.dump((data_digest, kwargs_digest, code_digest, samples), f, protocol=2)

        if method == 'mcmc':
            # write various meta info
            with open(diagnostics_filename, 'w') as f:
                f.write(f'seed: {samples.seed}\n')
                f.write(f'elapsed_time: {samples.elapsed_time}\n')
                f.write(f'check_treedepth: {samples.check_treedepth}\n')
                f.write(f'check_divergences: {samples.check_div}\n\n')
                f.write(f'check_fit:\n{samples.check_fit}\n')

        # if no parameters supplied, get the list from the samples
        if params is None:
            params = list(samples.keys())

        # plot parameter traces
        if method == 'mcmc' or method == 'vb':
            pars = [p for p in params if len(samples[p].shape) < 3]
            fig = plt.figure(figsize=(12, 2 * len(pars) + 3))
            gs = gridspec.GridSpec(len(pars) + 2 * (method == 'mcmc'), 2, width_ratios=[5, 1])
            for i, k in enumerate(pars):
                print(f'plotting {k}... ', end='', flush=True)
                ax = fig.add_subplot(gs[i, 0])
                ax.set_ylabel(k)
                is_multi = len(samples[k].shape) > 1
                if np.prod(samples[k].shape) > 1e6:
                    print('skipping', flush=True)
                    continue
                ax.plot(samples[k], lw=1, alpha=0.7 if is_multi else 0.9)
                ax = fig.add_subplot(gs[i, 1])
                if is_multi:
                    for j in range(samples[k].shape[1]):
                        ax.hist(samples[k][:, j], bins=30, alpha=0.8)
                else:
                    ax.hist(samples[k], bins=50)
                print('done', flush=True)

            if method == 'mcmc':

                # plot n_eff
                print(f'plotting n_eff', flush=True)
                ax = fig.add_subplot(gs[-2, :])
                ax.hist(samples.n_eff[np.isfinite(samples.n_eff)], bins=100)
                ax.set_xlabel('Number of effective samples')
                ax.set_ylabel('Param Count')

                # plot Rhat
                print(f'plotting Rhat', flush=True)
                ax = fig.add_subplot(gs[-1, :])
                ax.hist(samples.rhat[np.isfinite(samples.rhat)], bins=100)
                ax.set_xlabel('Rhat')
                ax.set_ylabel('Param Count')

            print(f'saving... ', end='', flush=True)
            fig.tight_layout()
            fig.savefig(traces_filename, dpi=150)
            plt.close(fig)
            print(f'done', flush=True)

        if method == 'mcmc':
            print(f'min, max Rhat = {np.nanmin(samples.rhat)}, {np.nanmax(samples.rhat)}')

    return samples


def pystan_vb_extract(results):
    param_specs = results['sampler_param_names']
    samples = results['sampler_params']
    n = len(samples[0])

    # first pass, calculate the shape
    param_shapes = OrderedDict()
    for param_spec in param_specs:
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            idxs = [int(i) for i in splt[1][:-1].split(',')]  # no +1 for shape calculation because pystan already returns 1-based indexes for vb!
        else:
            idxs = ()
        param_shapes[name] = np.maximum(idxs, param_shapes.get(name, idxs))

    # create arrays
    params = OrderedDict([(name, np.nan * np.empty((n, ) + tuple(shape))) for name, shape in param_shapes.items()])

    # second pass, set arrays
    for param_spec, param_samples in zip(param_specs, samples):
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            idxs = [int(i) - 1 for i in splt[1][:-1].split(',')]  # -1 because pystan returns 1-based indexes for vb!
        else:
            idxs = ()
        params[name][(..., ) + tuple(idxs)] = param_samples

    return params


def _hash_dict_with_numpy_arrays(d):
    m = md5()
    for k, v in sorted(d.items()):
        m.update(pickle.dumps(k))
        if type(v) == np.ndarray:
            m.update(v.tobytes())
        else:
            m.update(pickle.dumps(v))
    return m.hexdigest()
