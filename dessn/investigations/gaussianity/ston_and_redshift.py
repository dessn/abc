import numpy as np
import os
import itertools
from joblib import Parallel, delayed
import sncosmo
from astropy.table import Table
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

from dessn.chain.chain import ChainConsumer
from dessn.investigations.gaussianity.apparent_mags import generate_and_return
from dessn.investigations.gaussianity.model import PerfectRedshift
from dessn.framework.samplers.ensemble import EnsembleSampler


def realise_light_curve(seed):
    np.random.seed(seed)
    t0 = 1000
    x0 = 1e-5
    x1 = 0
    c = 0
    z = np.random.uniform(0.1, 0.9)

    num_obs = 20
    deltat = -35
    ts = np.arange(t0 + deltat, (t0 + deltat) + 5 * num_obs, 5)

    times = np.array([[t, t + 0.05, t + 0.1, t + 0.2] for t in ts]).flatten()
    bands = [b for t in ts for b in ['desg', 'desr', 'desi', 'desz']]
    gains = np.ones(times.shape)
    skynoise = np.random.uniform(low=50, high=300) * np.ones(times.shape)
    zp = 30 * np.ones(times.shape)
    zpsys = ['ab'] * times.size

    obs = Table({'time': times,
                 'band': bands,
                 'gain': gains,
                 'skynoise': skynoise,
                 'zp': zp,
                 'zpsys': zpsys})

    model = sncosmo.Model(source='salt2-extended')
    p = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}
    model.set(z=z)
    lc = sncosmo.realize_lcs(obs, model, [p])[0]
    ston = (lc["flux"] / lc["fluxerr"]).max()
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
        weights = None
    elif type == "mcmc":
        res, fitted_model = sncosmo.mcmc_lc(lc, model, ['t0', 'x0', 'x1', 'c'], nburn=500, nwalkers=20,
                                            nsamples=2000, guess_amplitude=False, guess_t0=False)
        chain = res.samples
        weights = None
    else:
        bounds = {"t0": [980, 1020], "x0": [0.5e-5, 1.5e-5], "x1": [-10, 10], "c": [-2, 2]}
        res, fitted_model = sncosmo.nest_lc(lc, model, ['t0', 'x0', 'x1', 'c'], bounds, npoints=1000,
                                            guess_amplitude=False, guess_t0=False)
        chain = res.samples
        weights = res.weights

    # fig = sncosmo.plot_lc(lc, model=[fitted_model, correct_model], errors=res.errors)
    # fig.savefig(temp_dir + os.sep + "lc_%d.png" % seed, bbox_inches="tight", dpi=300)
    map = {"x0": "$x_0$", "x1": "$x_1$", "c": "$c$", "t0": "$t_0$"}
    parameters = [map[a] for a in res.vparam_names]

    chain, parameters = add_mu_to_chain(interped, chain, parameters)
    return chain, parameters, weights, res.parameters[1:], res.covariance


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


def plot_results(chain, param, chainf, paramf, w, t0, x0, x1, c, temp_dir, seed, interped):
    cc = ChainConsumer()
    cc.add_chain(chain, parameters=param, name="Posterior")
    cc.add_chain(chainf, parameters=paramf, name="Gaussian", weights=w)
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


def get_result(temp_dir, seed, method="iminuit"):
    save_file = temp_dir + "/final%d.npy" % seed

    if os.path.exists(save_file):
        res = np.load(save_file)
        if res.size == 0:
            return None
        else:
            print(seed, res[-2], res[-1])
            return np.concatenate(([seed], res))
    else:
        interp = generate_and_return()
        z, t0, x0, x1, c, ston, lc = realise_light_curve(seed)

        if ston < 5 or ston > 10:
            np.save(save_file, np.array([]))
            return None

        try:
            chainf, parametersf, weights, means, cov = get_gaussian_fit(z, t0, x0, x1, c, lc, seed,
                                                               temp_dir, interp, type=method)
            if method in ["iminuit"]:
                chain, parameters = get_posterior(z, t0, lc, seed, temp_dir, interp)
                weights = None
            else:
                chain = chainf
                parameters = parametersf
                chainf = np.random.multivariate_normal(means, cov, size=int(1e6))
                chainf, _ = add_mu_to_chain(interp, chainf, parameters[:-1])

            assert is_fit_good(t0, x0, x1, c, means, cov)
            assert not is_unconstrained(chain, parameters)

        except Exception as e:
            print(e)
            np.save(save_file, np.array([]))
            return None

        plot_results(chain, parameters, chainf, parametersf, weights, t0, x0, x1, c, temp_dir, seed, interp)

        muf = chainf[:, -1]
        mu = chain[:, -1]

        mean2 = np.mean(chain[:, :-1], axis=0)
        cov2 = np.cov(chain[:, :-1].T)
        chain2 = np.random.multivariate_normal(mean2, cov2, size=int(1e6))
        chain2, _ = add_mu_to_chain(interp, chain2, parameters[:-1])
        mu2 = chain2[:, -1]

        diffmu = np.average(muf) - np.average(mu, weights=weights)
        diffmu2 = np.average(mu2) - np.average(mu, weights=weights)
        diffstd = np.std(muf) / np.std(mu)
        diffstd2 = np.std(mu2) / np.std(mu)
        res = np.array([z, t0, x0, x1, c, ston, diffmu, diffstd, diffmu2, diffstd2])
        np.save(save_file, res)
        return np.concatenate(([seed], res))


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

    temp_dir = os.path.dirname(__file__) + "/output/randoms"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    n = 2500
    res = Parallel(n_jobs=1, max_nbytes="20M", verbose=100, batch_size=1)(delayed(get_result)(
        temp_dir, i) for i in range(n))
    res = np.array([r for r in res if r is not None])
    s = res[:, 6]
    res = res[(s > 5) & (s < 10), :]

    seeds = res[:, 0]
    z = res[:, 1]
    s = res[:, 6]
    diffmu2 = res[:, -2]
    diffstd2 = res[:, -1]
    diffmu = res[:, -4]
    diffstd = res[:, -3]

    diffstd = 100 * (diffstd - 1)
    diffstd2 = 100 * (diffstd2 - 1)

    import matplotlib.pyplot as plt

    zs = np.linspace(z.min(), z.max(), 50)
    ston = np.linspace(s.min(), s.max(), 50)

    xx, yy = np.meshgrid(zs, ston, indexing='ij')

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))

    axes = axes.flatten()
    for ax, data in zip(axes, [diffmu, diffstd, diffmu2, diffstd2]):
        if True:
            m = polyfit2d(z, s, data, order=4)
            zz = polyval2d(xx, yy, m)
        else:
            zz = griddata((z, s), data, (xx, yy), method="nearest")

        if np.min(zz) > 0.0:
            vmax = np.max(np.abs(zz - 1)) + 1
            vmin = 1 - (vmax - 1)
        else:
            vmax = np.max(np.abs(zz))
            vmin = -vmax
        h = ax.contourf(xx, yy, zz, 30, cmap='bwr', vmin=vmin, vmax=vmax)
        ax.scatter(z, s, c=data, s=20, cmap='bwr', vmin=vmin, vmax=vmax)
        ax.set_ylim(5, 10)
        ax.set_xlim(z.min(), z.max())
        div1 = make_axes_locatable(ax)
        cax1 = div1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(h, cax=cax1)
        if False:
            for i, zp, sp in zip(seeds, z, s):
                ax.text(zp, sp, "%d" % i, alpha=0.3)
        ax.set_xlabel("$z$")
        ax.set_ylabel("$S/N$")
    plt.tight_layout()
    fig.savefig(os.path.dirname(__file__) + "/output/bias.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()
