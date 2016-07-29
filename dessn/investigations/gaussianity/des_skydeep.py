import numpy as np
import os
import itertools
from joblib import Parallel, delayed
import sncosmo
from astropy.table import Table
from scipy.stats import skew

from dessn.chain.chain import ChainConsumer
from dessn.investigations.gaussianity.apparent_mags import generate_and_return
from dessn.investigations.gaussianity.model import PerfectRedshift
from dessn.framework.samplers.ensemble import EnsembleSampler


def realise_light_curve(temp_dir, seed):
    np.random.seed(seed)
    t0 = 1000
    num_obs = 20
    z = np.random.uniform(0.1, 0.9)
    deltat = -35

    newmoon = 1000 - np.random.uniform(0, 29.5)

    ts = np.arange(t0 + deltat, (t0 + deltat) + 5 * num_obs, 5)

    bands = [b for t in ts for b in ['desg', 'desr', 'desi', 'desz']]
    zps = np.array([b for t in ts for b in [34.24, 34.85, 34.94, 35.42]])  # Deep field
    mins = np.array([b for t in ts for b in [22.1, 21.1, 20.1, 18.7]])
    maxs = np.array([b for t in ts for b in [19.4, 19.7, 19.4, 18.2]])
    seeing = np.array([b for t in ts for b in [1.06, 1.00, 0.96, 0.93]])
    times = np.array([[t, t + 0.05, t + 0.1, t + 0.2] for t in ts]).flatten()

    full = 0.5 + 0.5 * np.sin((times - newmoon) * 2 * np.pi / 29.5)
    perm = np.random.uniform(-0.1, 0.1, full.shape)
    seeing2 = np.random.uniform(4, 6, full.shape)
    sigma_psf = seeing2 / 2.36

    sky_noise = np.array([np.sqrt(10.0**(((maxx - minn) * f + minn + p - zp) / -2.5) * 0.263**2) *
                          np.sqrt(4 * np.pi) * s
                          for f, p, s, minn, maxx, zp in zip(full, perm, sigma_psf, mins, maxs, zps)])

    # sky_noise = np.array([np.sqrt(10.0**(((maxx - minn) * f + minn + p - zp) / -2.5)) *
    #                       np.pi * ((2.04 * se) / 2) ** 2
    #                       for f,p,minn,maxx,zp,se in zip(full, perm, mins, maxs, zps, seeing)])

    zpsys = ['ab'] * times.size
    gains = np.ones(times.shape)

    obs = Table({'time': times,
                 'band': bands,
                 'gain': gains,
                 'skynoise': sky_noise,
                 'zp': zps,
                 'zpsys': zpsys})

    model = sncosmo.Model(source='salt2-extended')
    model.set(z=z)
    mabs = np.random.normal(-19.3, 0.3)
    model.set_source_peakabsmag(mabs, 'bessellb', 'ab')
    x0 = model.get('x0')
    x1 = 0
    c = 0
    p = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}

    lc = sncosmo.realize_lcs(obs, model, [p])[0]
    fig = sncosmo.plot_lc(lc)
    fig.savefig(temp_dir + os.sep + "lc_%d.png" % seed, bbox_inches="tight", dpi=300)
    ston = (lc["flux"] / lc["fluxerr"]).max()
    print(z, t0, x0, x1, c, ston)
    return z, t0, x0, x1, c, ston, lc


def get_gaussian_fit(z, t0, x0, x1, c, lc, seed, temp_dir, interped, type="iminuit"):
    model = sncosmo.Model(source='salt2-extended')
    p = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}
    model.set(**p)

    correct_model = sncosmo.Model(source='salt2-extended')
    correct_model.set(**p)
    if type == "iminuit":
        res, fitted_model = sncosmo.fit_lc(lc, model, ['t0', 'x0', 'x1', 'c'],
                                           guess_amplitude=False, guess_t0=False)
        chain = np.random.multivariate_normal(res.parameters[1:], res.covariance, size=int(1e5))
        fig = sncosmo.plot_lc(lc, model=[fitted_model, correct_model], errors=res.errors)
        fig.savefig(temp_dir + os.sep + "lc_%d.png" % seed, bbox_inches="tight", dpi=300)
    elif type == "mcmc":
        res, fitted_model = sncosmo.mcmc_lc(lc, model, ['t0', 'x0', 'x1', 'c'], nburn=500, nwalkers=20,
                                            nsamples=1500, guess_amplitude=False, guess_t0=False)
        chain = np.random.multivariate_normal(res.parameters[1:], res.covariance, size=int(1e5))
    elif type == "nestle":
        bounds = {"t0": [980, 1020], "x0": [0.1e-6, 9e-3], "x1": [-10, 10], "c": [-1, 1]}
        res, fitted_model = sncosmo.nest_lc(lc, model, ['t0', 'x0', 'x1', 'c'], bounds,
                                            guess_amplitude=False, guess_t0=False)
        chain = np.random.multivariate_normal(res.parameters[1:], res.covariance, size=int(1e5))
    else:
        raise ValueError("Type %s not recognised" % type)

    map = {"x0": "$x_0$", "x1": "$x_1$", "c": "$c$", "t0": "$t_0$"}
    parameters = [map[a] for a in res.vparam_names]

    chain, parameters = add_mu_to_chain(interped, chain, parameters)
    return chain, parameters, res.parameters[1:], res.covariance


def get_posterior(z, t0, lc, seed, temp_dir, interped):
    my_model = PerfectRedshift([lc], [z], t0, name="posterior%d" % seed)
    sampler = EnsembleSampler(temp_dir=temp_dir, num_burn=500, num_steps=1500)
    c = my_model.fit(sampler)
    chain = c.chains[-1]
    parameters = c.parameters[-1]

    chain, parameters = add_mu_to_chain(interped, chain, parameters)
    return chain, parameters


def is_fit_good(t0, x0, x1, c, means, cov):
    t0f, x0f, x1f, cf = tuple(means)
    t0fc, x0fc, x1fc, cfc = tuple(np.sqrt(np.diag(cov)).tolist())
    isgood = np.abs(t0 - t0f) < 20 and x0f > 0.0 and np.abs(x1 - x1f) < 4 and np.abs(c - cf) < 2
    isgood = isgood and t0fc < 20 and x0fc < 1e-5 and x1fc < 4 and cfc < 1
    return isgood


def get_mu(interped, x0, x1, c):
    alpha = 0.14
    beta = 3.15
    mu = interped(x1, c, grid=False) - (0.4 * np.log10(np.abs(x0) / 1e-5)) + alpha * x1 - beta * c
    return mu


def add_mu_to_chain(interped, chain, parameters):
    x0s = chain[:, parameters.index("$x_0$")]
    x1s = chain[:, parameters.index("$x_1$")]
    cs = chain[:, parameters.index("$c$")]
    mu = get_mu(interped, x0s, x1s, cs)
    chain = np.hstack((chain, mu[:, None]))
    return chain, parameters + [r"$\mu$"]


def plot_results(chain, param, chainf, chainf2, chainf3, paramf, t0, x0, x1, c, temp_dir, seed, interped):
    cc = ChainConsumer()
    cc.add_chain(chain, parameters=param, name="Posterior")
    cc.add_chain(chainf, parameters=paramf, name="Minuit")
    cc.add_chain(chainf2, parameters=paramf, name="Emcee")
    cc.add_chain(chainf3, parameters=paramf, name="Nestle")
    truth = {"$t_0$": t0, "$x_0$": x0, "$x_1$": x1, "$c$": c, r"$\mu$": get_mu(interped, x0, x1, c)}
    cc.plot(filename=temp_dir + "/surfaces_%d.png" % seed, truth=truth)


def is_unconstrained(chain, param):
    c = ChainConsumer()
    c.add_chain(chain, parameters=param)
    constraints = c.get_summary()[0]
    for key in constraints:
        val = constraints[key]
        if val[0] is None or val[2] is None:
            return True
    return False


def get_result(temp_dir, seed):
    save_file = temp_dir + "/final%d.npy" % seed

    if os.path.exists(save_file):
        res = np.load(save_file)
        if res.size == 0:
            return None
        else:
            return res
    else:
        interp = generate_and_return()
        z, t0, x0, x1, c, ston, lc = realise_light_curve(temp_dir, seed)

        if ston < 2:
            np.save(save_file, np.array([]))
            return None

        try:
            chainf, parametersf, means, cov = get_gaussian_fit(z, t0, x0, x1, c, lc, seed,
                                                               temp_dir, interp, type="iminuit")

            mcmc_file = temp_dir + "/mcmc%d.npy" % seed
            if os.path.exists(mcmc_file):
                chainf2 = np.load(mcmc_file)
            else:
                chainf2, _, _, _ = get_gaussian_fit(z, t0, x0, x1, c, lc, seed, temp_dir, interp, type="mcmc")
                np.save(mcmc_file, chainf2)

            nestle_file = temp_dir + "/nestle%d.npy" % seed
            if os.path.exists(nestle_file):
                chainf3 = np.load(nestle_file)
            else:
                chainf3, _, _, _ = get_gaussian_fit(z, t0, x0, x1, c, lc, seed, temp_dir, interp, type="nestle")
                np.save(nestle_file, chainf3)
            chain, parameters = get_posterior(z, t0, lc, seed, temp_dir, interp)

            assert is_fit_good(t0, x0, x1, c, means, cov)
            assert not is_unconstrained(chain, parameters)

        except Exception as e:
            print(e)
            np.save(save_file, np.array([]))
            return None

        try:
            plot_results(chain, parameters, chainf, chainf2, chainf3, parametersf, t0, x0, x1, c, temp_dir, seed, interp)
        except Exception as e:
            pass

        muf = chainf[:, -1]
        muf2 = chainf2[:, -1]
        muf3 = chainf3[:, -1]
        mu = chain[:, -1]

        mean2 = np.mean(chain[:, :-1], axis=0)
        cov2 = np.cov(chain[:, :-1].T)
        chain2 = np.random.multivariate_normal(mean2, cov2, size=int(1e6))
        chain2, _ = add_mu_to_chain(interp, chain2, parameters[:-1])
        mu2 = chain2[:, -1]

        mus = [np.mean(mu), np.mean(mu2), np.mean(muf), np.mean(muf2), np.mean(muf3)]
        stds = [np.std(mu), np.std(mu2), np.std(muf), np.std(muf2), np.std(muf3)]
        res = np.array([seed, z, t0, x0, x1, c, ston] + mus + stds + [skew(mu)])
        np.save(save_file, res)
        return res


def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m


def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

if __name__ == "__main__":

    temp_dir = os.path.dirname(__file__) + "/output/dessky_deep"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    n = 210
    res = Parallel(n_jobs=4, max_nbytes="20M", verbose=100, batch_size=1)(delayed(get_result)(
        temp_dir, i) for i in range(n))
    res = np.array([r for r in res if r is not None])
    print(res)
    s = res[:, 6]
    res = res[(s > 5), :]

    seeds = res[:, 0]
    z = res[:, 1]
    s = res[:, 6] / 100
    skews = res[:, -1]

    diff_mu_ps = -res[:, 7] + res[:, 8]
    diff_mu_fs = -res[:, 7] + res[:, 9]
    diff_mu_ms = -res[:, 7] + res[:, 10]
    diff_mu_ns = -res[:, 7] + res[:, 11]

    stdd = res[:, 12]

    diff_std_ps = res[:, 12] / res[:, 13]
    diff_std_fs = res[:, 12] / res[:, 14]
    diff_std_ms = res[:, 12] / res[:, 15]
    diff_std_ns = res[:, 12] / res[:, 16]

    diff_std_ps2 = 100 * (diff_std_ps - 1)
    diff_std_fs2 = 100 * (diff_std_fs - 1)
    diff_std_ms2 = 100 * (diff_std_ms - 1)
    diff_std_ns2 = 100 * (diff_std_ns - 1)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 7))

    axes = axes.T.flatten()
    datas = [[diff_mu_ps, diff_mu_fs, diff_mu_ms, diff_mu_ns],
             [diff_mu_ps/stdd, diff_mu_fs/stdd, diff_mu_ms/stdd, diff_mu_ns/stdd],
             [diff_std_ps2, diff_std_fs2, diff_std_ms2, diff_std_ns2]]
    labels = [["Posterior", "minuit", "emcee", "nestle"], ["Posterior", "minuit", "emcee", "nestle"], ["P/PSS", "P/FSS", "P/MSS", "P/NSS"]]

    axes[0].set_ylabel(r"$\Delta (\mu + M)$", fontsize=14)
    axes[1].set_ylabel(r"$\Delta (\mu + M) / \sigma_{\mu + M}$", fontsize=14)
    axes[2].set_ylabel(r"Percent Ratio $\Delta \sigma_{\mu + M}$", fontsize=14)
    zs = np.linspace(z.min(), z.max(), 11)

    for ax, data, label in zip(axes, datas, labels):
        ax.set_xlabel("$z$", fontsize=14)
        for i, (d, l) in enumerate(zip(data, label)):
            n, _ = np.histogram(z, bins=zs)
            mean, edges = np.histogram(z, bins=zs, weights=d)
            std, _ = np.histogram(z, bins=zs, weights=d*d)
            mean /= n
            std = np.sqrt(std / n - mean * mean)
            ec = 0.5 * (edges[:-1] + edges[1:])
            ax.errorbar(ec + (0.01 * (i - 1)), mean, fmt='o', yerr=std, label=l)
    axes[0].legend(loc=2)
    skewhist, edges = np.histogram(z, bins=zs, weights=skews)
    std, _ = np.histogram(z, bins=zs, weights=skews * skews)
    skewhist /= n
    std = np.sqrt(std / n - skewhist * skewhist)
    ec = 0.5 * (edges[:-1] + edges[1:])
    axes[-1].errorbar(ec, skewhist, fmt='o', yerr=std, label="Skewness")
    axes[-1].legend(loc=3)
    axes[-1].set_xlabel("$z$")

    plt.tight_layout()
    fig.savefig(os.path.dirname(__file__) + "/output/bias_dessky_deep.png", dpi=300, bbox_inches="tight", transparent=True)
    # plt.show()
