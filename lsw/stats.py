import numpy as np
import scipy.stats
from scipy.special import gammaln


def rhat_orig(thetas, m, axis=0):
    """
    Vehtari, A.; Gelman, A.; Simpson, D.; Carpenter, B. & Bürkner, P.-C.
    Rank-normalization, folding, and localization: An improved Rhat for assessing convergence of MCMC.
    arXiv preprint arXiv:1903.08008, 2019
    :param thetas: an array of values for parameter, length must be divisible by n
    :param m: number of chains, or supply double the number of chains for split-rhat
    :param axis: the axis of theta corresponding to samples, default 0
    :return: non-rank-normalised rhat, for rank-normalised version see the rhat_ranknorm function
    """

    def rhat_1d(theta):
        theta = np.reshape(theta, (-1, m), order='F')  # (n, m) with n draws per chain, and m chains
        n = theta.shape[0]

        var_between_chain = theta.mean(axis=0).var(ddof=1)
        var_within_chain = theta.var(ddof=1, axis=0).mean()
        var = (n-1)/n * var_within_chain + var_between_chain

        return np.sqrt(var / var_within_chain)

    return np.apply_along_axis(rhat_1d, axis, thetas)


def rhat_ranknorm(thetas, m, axis=0):
    """
    Vehtari, A.; Gelman, A.; Simpson, D.; Carpenter, B. & Bürkner, P.-C.
    Rank-normalization, folding, and localization: An improved Rhat for assessing convergence of MCMC.
    arXiv preprint arXiv:1903.08008, 2019
    :param thetas: 1D array of values for parameter, length must be divisible by n
    :param m: number of chains, or supply double the number of chains for split-rhat
    :param axis: the axis of theta corresponding to samples, default 0
    :return: rank-normalised rhat, for non-rank-normalised version see the rhat function
    """

    def rhat_ranknown_1d(theta):
        r = scipy.stats.rankdata(theta)
        z = scipy.stats.norm.ppf((r - 0.5)/len(r))
        return rhat_orig(z, m)

    return np.apply_along_axis(rhat_ranknown_1d, axis, thetas)


def rhat(thetas, m, axis=0):
    """
    Vehtari, A.; Gelman, A.; Simpson, D.; Carpenter, B. & Bürkner, P.-C.
    Rank-normalization, folding, and localization: An improved Rhat for assessing convergence of MCMC.
    arXiv preprint arXiv:1903.08008, 2019
    :param thetas: 1D array of values for parameter, length must be divisible by n
    :param m: number of chains, or supply double the number of chains for split-rhat
    :param axis: the axis of theta corresponding to samples, default 0
    :return: maximum of the rank-normalised rhat and folded rank-normalised rhat (section 4.2 of paper)
    """

    zetas = np.abs(thetas - np.median(thetas, axis=axis, keepdims=True))
    rhat_folded = rhat_ranknorm(zetas, m, axis)
    rhat_unfolded = rhat_ranknorm(thetas, m, axis)

    return np.maximum(rhat_folded, rhat_unfolded)


def entropy_discrete(p, axis=None):
    return -np.sum(p * np.log2(p), axis=axis)


def jensen_shannon_div_norm(mu, sigma, w=None):
    """
    Generalised Jensen-Shannon divergence from equation (5.1) of the paper
    Lin, J. Divergence measures based on the Shannon entropy IEEE Transactions on Information theory, IEEE, 1991, 37, 145-151.
    :param mu: n location or mean parameters
    :param sigma: n scale or standard deviation parameters
    :param w: n weights, or None if all are equal
    :return: float containing the divergence measure
    """

    n = len(mu)
    mu = np.array(mu)
    sigma = np.array(sigma)

    # make weight matrix sum to 1
    if w is None:
        w = np.ones(n)
    w = (np.array(w) / np.sum(w)).reshape((1, n))

    mu_sum = np.dot(w, mu)
    sigma_sum = np.sqrt(np.dot(w, sigma**2))
    entropy_of_sum = scipy.stats.norm(mu_sum, sigma_sum).entropy()

    sum_of_entropies = np.dot(w, scipy.stats.norm(mu, sigma).entropy())

    return entropy_of_sum - sum_of_entropies


def jensen_shannon_div_bern(p, w=None):
    """
    Generalised Jensen-Shannon divergence from equation (5.1) of the paper
    Lin, J. Divergence measures based on the Shannon entropy IEEE Transactions on Information theory, IEEE, 1991, 37, 145-151.
    :param p: n probabilities for the Bernoulli distribution
    :param w: n weights, or None if all are equal
    :return: float containing the divergence measure
    """

    n = len(p)
    p = np.array(p)

    # make weight matrix sum to 1
    if w is None:
        w = np.ones(n)
    w = (np.array(w) / np.sum(w)).reshape((1, n))

    p_sum = np.dot(w, p)
    entropy_of_sum = scipy.stats.bernoulli(p_sum).entropy()

    sum_of_entropies = np.dot(w, scipy.stats.bernoulli(p).entropy())

    return entropy_of_sum - sum_of_entropies


def jensen_shannon_div(p, w=None):
    """
    Generalised Jensen-Shannon divergence from equation (5.1) of the paper
    Lin, J. Divergence measures based on the Shannon entropy IEEE Transactions on Information theory, IEEE, 1991, 37, 145-151.
    :param p: (n,k) matrix of probabilities for n discrete distributions, each k values, s.t. for any distribution i, np.sum(p[i,:]) == 1
    :param w: n weights, or None if all are equal
    :return: float containing the divergence measure
    """

    n, k = p.shape

    # make weight matrix sum to 1
    if w is None:
        w = np.ones(n)
    w = (np.array(w) / np.sum(w)).reshape((1, n))

    # weighted-average of the probability distributions
    p_avg = (w @ p)[0]

    return entropy_discrete(p_avg) - np.sum(w * entropy_discrete(p, axis=1))


# pretty much copied (with minor changes) from scikits.bootstrap which wouldn't install on my system
# see https://github.com/cgevans/scikits-bootstrap/blob/master/scikits/bootstrap/bootstrap.py#L20
def bootci_pi(data, statfunc=lambda x: np.mean(x, axis = 0), alpha=0.05, n_samples=10000):
    data = np.array(data)
    bootindexes = (np.random.randint(0, data.shape[0], size=(data.shape[0],)) for _ in range(n_samples))
    stat = np.array([statfunc(data[i]) for i in bootindexes])
    stat.sort(axis=0)
    alphas = np.array([0.5 * alpha, 1 - 0.5 * alpha])
    return stat[np.round((n_samples-1)*alphas).astype('int')]


def pct_from_sigma(sigma):
    return 100 * (scipy.stats.norm.cdf(sigma) - scipy.stats.norm.cdf(-sigma))


def ci_hdi(x, pct=95, sigma=None):

    # sigma overrides pct
    if sigma is not None:
        pct = pct_from_sigma(sigma)

    x = np.sort(x)

    ci_idx_inc = int(np.floor(0.01 * pct * len(x)))
    n_cis = len(x) - ci_idx_inc

    ci_width = np.zeros(n_cis)
    for i in range(n_cis):
        ci_width[i] = x[i + ci_idx_inc] - x[i]

    min_width_idx = np.argmin(ci_width)
    return x[min_width_idx], x[min_width_idx + ci_idx_inc]


def ci_pctl(x, pct=95, **kwargs):
    x_lower = np.percentile(x, 0.5 * (100 - pct), **kwargs)
    x_upper = np.percentile(x, 0.5 * (100 + pct), **kwargs)
    return x_lower, x_upper


def cov_ellipse(x, nstd=2):

    pos = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)

    return pos, width, height, theta


def make_cov(sd, rho):
    cov = np.zeros((2, 2))
    cov[0, 0] = sd[0] ** 2
    cov[0, 1] = sd[0] * sd[1] * rho
    cov[1, 0] = cov[0, 1]
    cov[1, 1] = sd[1] ** 2
    return cov


# Easily accesible source:
# Wang, F. & Gelfand, A. E.
# Directional data analysis under the general projected normal distribution.
# Statistical methodology, Elsevier, 2013, 10, 113-127.
def projected_normal_pdf(theta, mu, sd, rho):

    cov = np.zeros((2, 2))
    cov[0, 0] = sd[0] ** 2
    cov[0, 1] = sd[0] * sd[1] * rho
    cov[1, 0] = cov[0, 1]
    cov[1, 1] = sd[1] ** 2

    ct = np.cos(theta)
    st = np.sin(theta)

    phi = scipy.stats.multivariate_normal.pdf(mu, cov=cov)
    a = 1.0 / (sd[0] * sd[1] * np.sqrt(1 - rho**2))
    c = a**2 * (cov[1, 1] * ct ** 2 - 2 * rho * cov[0, 1] * ct * st + cov[0, 0] * st ** 2)
    d = a**2 / np.sqrt(c) * (mu[0] * sd[1] * (sd[1] * ct - rho * sd[0] * st) +
                             mu[1] * sd[0] * (sd[0] * st - rho * sd[1] * ct))

    pdf_param = a / np.sqrt(c) * (mu[0] * st - mu[1] * ct)
    return (phi + a * d * scipy.stats.norm.cdf(d) * scipy.stats.norm.pdf(pdf_param)) / c


def beta_binomial_lpmf(n, N, alpha, beta):
    num = gammaln(N + 1) + gammaln(n + alpha) + gammaln(N - n + beta) + gammaln(alpha + beta)
    den = gammaln(n + 1) + gammaln(N - n + 1) + gammaln(N + alpha + beta) + gammaln(alpha) + gammaln(beta)
    return num - den
