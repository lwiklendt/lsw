import numpy as np
import scipy.stats


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


def ci_pctl(x, pct=95):
    x_lower = np.percentile(x, 0.5 * (100 - pct))
    x_upper = np.percentile(x, 0.5 * (100 + pct))
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
