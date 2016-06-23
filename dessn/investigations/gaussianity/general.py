from dessn.framework.samplers.ensemble import EnsembleSampler
from dessn.investigations.gaussianity.apparent_mags import generate_and_return
import sncosmo
import numpy as np
from astropy.table import Table
from dessn.chain.chain import ChainConsumer
import os
from joblib import Parallel, delayed
from scipy.interpolate import griddata

from dessn.investigations.gaussianity.model import PerfectRedshift


def get_mu_from_chain(interp, chain, params):
    alpha = 0.14
    beta = 3.15
    x0s = chain[:, params.index("$x_0$")]
    x1s = chain[:, params.index("$x_1$")]
    cs = chain[:, params.index("$c$")]
    mu = interp(x1s, cs, grid=False) - (0.4 * np.log10(np.abs(x0s) / 1e-5)) + alpha * x1s - beta * cs
    return mu


def random_obs(temp_dir, seed):
    np.random.seed(seed)
    interp = generate_and_return()
    x1 = np.random.normal()
    # colour = np.random.normal(scale=0.1)
    colour = 0
    x0 = 1e-5
    # t0 = np.random.uniform(low=1000, high=2000)
    t0 = 1000
    z = np.random.uniform(low=0.1, high=1.0)

    # deltat = np.random.uniform(low=-20, high=0)
    # num_obs = np.random.randint(low=10, high=40)
    num_obs = 20
    deltat = -35

    filename = temp_dir + "/save_%d.npy" % seed

    if not os.path.exists(filename):
        ts = np.arange(t0 + deltat, (t0 + deltat) + 5 * num_obs, 5)

        times = np.array([[t, t + 0.05, t + 0.1, t + 0.2] for t in ts]).flatten()
        bands = [b for t in ts for b in ['desg', 'desr', 'desi', 'desz']]
        gains = np.ones(times.shape)
        skynoise = np.random.uniform(low=20, high=800) * np.ones(times.shape)
        zp = 30 * np.ones(times.shape)
        zpsys = ['ab'] * times.size

        obs = Table({'time': times,
                     'band': bands,
                     'gain': gains,
                     'skynoise': skynoise,
                     'zp': zp,
                     'zpsys': zpsys})
        model = sncosmo.Model(source='salt2')
        p = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': colour}
        model.set(z=z)
        print(seed, " Vals are ", p)
        lc = sncosmo.realize_lcs(obs, model, [p])[0]
        ston = (lc["flux"] / lc["fluxerr"]).max()

        model.set(t0=t0, x1=x1, c=colour, x0=x0)
        try:
            res, fitted_model = sncosmo.fit_lc(lc, model, ['t0', 'x0', 'x1', 'c'],
                                                guess_amplitude=False, guess_t0=False)
        except ValueError:
            return np.nan, np.nan, x1, colour, num_obs, ston, deltat, z, 0

        fig = sncosmo.plot_lc(lc, model=fitted_model, errors=res.errors)
        fig.savefig(temp_dir + os.sep + "lc_%d.png" % seed, bbox_inches="tight", dpi=300)
        my_model = PerfectRedshift([lc], [z], t0, name="posterior%d" % seed)
        sampler = EnsembleSampler(temp_dir=temp_dir, num_burn=400, num_steps=1500)
        c = ChainConsumer()
        my_model.fit(sampler, chain_consumer=c)
        map = {"x0": "$x_0$", "x1": "$x_1$", "c": "$c$", "t0": "$t_0$"}
        parameters = [map[a] for a in res.vparam_names]

        mu1 = get_mu_from_chain(interped, c.chains[-1], c.parameters[-1])
        c.parameteers[-1].append(r"$\mu$")
        c.chains[-1] = np.hstack((c.chains[-1], mu1[:, None]))

        chain2 = np.random.multivariate_normal(res.parameters[1:], res.covariance, size=int(1e5))
        chain2 = np.hstack((chain2, get_mu_from_chain(interp, chain2, parameters)[:, None]))
        c.add_chain(chain2, parameters=parameters, name="Gaussian")
        figfilename = filename.replace(".npy", ".png")
        c.plot(filename=figfilename,
               truth={"$t_0$": t0, "$x_0$": x0, "$x_1$": x1, "$c$": colour})

        means = []
        stds = []
        isgood = (np.abs(x1 - res.parameters[3]) < 4) & (np.abs(colour - res.parameters[4]) < 2) & \
                 (res.parameters[2] > 0.0)
        isgood *= 1.0

        for i in range(len(c.chains)):
            a = c.chains[i][:, -1]
            means.append(a.mean())
            stds.append(np.std(a))
        diffmu = np.diff(means)[0]
        diffstd = np.diff(stds)[0]
        np.save(filename, np.array([diffmu, diffstd, ston, 1.0 * isgood]))

    else:
        vals = np.load(filename)
        diffmu = vals[0]
        diffstd = vals[1]
        ston = vals[2]
        isgood = vals[3]

    return diffmu, diffstd, x1, colour, num_obs, ston, deltat, z, isgood

if __name__ == "__main__":
    temp_dir = os.path.dirname(__file__) + "/output/randoms"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    n = 200
    res = Parallel(n_jobs=4, max_nbytes="20M", verbose=100, batch_size=1)(delayed(random_obs)(
        temp_dir, i) for i in range(n))
    res = np.array(res)

    good = np.isfinite(res[:, -1]) == 1
    res = res[good, :]
    print(res.shape)
    z = res[:, -1]
    s = res[:, 5]

    zs = np.linspace(z.min(), z.max(), 50)
    ston = np.linspace(s.min(), s.max(), 50)

    zz, ss = np.meshgrid(zs, ston, indexing='ij')
    interped = griddata((z, s), res[:, 0], (zz, ss), method="nearest")

    print(res[:, 0])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    h = ax.contourf(zz, ss, np.abs(interped), 30, cmap='viridis', vmin=0.0, vmax=0.5)
    ax.set_xlabel("$z$")
    ax.set_ylabel("$S/N$")
    ax.set_ylim(2, 10)
    ax.plot(res[:, -1], res[:, 5], 'k.', alpha=0.3)
    plt.colorbar(h)
    plt.show()
