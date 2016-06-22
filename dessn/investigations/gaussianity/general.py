from dessn.investigations.gaussianity.model import PerfectRedshift
from dessn.investigations.gaussianity.apparent_mags import generate_and_return
import sncosmo
import numpy as np
from astropy.table import Table
from dessn.chain.chain import ChainConsumer
from dessn.framework.samplers.ensemble import EnsembleSampler
import os
from joblib import Parallel, delayed


def random_obs(temp_dir, seed):
    np.random.seed(seed)
    x1 = np.random.normal()
    # colour = np.random.normal(scale=0.1)
    colour = 0
    x0 = 1e-5
    # t0 = np.random.uniform(low=1000, high=2000)
    t0 = 1000
    z = np.random.uniform(low=0.1, high=0.8)

    # deltat = np.random.uniform(low=-20, high=0)
    # num_obs = np.random.randint(low=10, high=40)
    num_obs = 20
    deltat = -35
    ts = np.arange(t0 + deltat, (t0 + deltat) + 5 * num_obs, 5)

    times = np.array([[t, t + 0.05, t + 0.1, t + 0.2] for t in ts]).flatten()
    bands = [b for t in ts for b in ['desg', 'desr', 'desi', 'desz']]
    gains = np.ones(times.shape)
    skynoise = np.random.uniform(low=20, high=400) * np.ones(times.shape)
    zp = 30 * np.ones(times.shape)
    zpsys = ['ab'] * times.size

    obs = Table({'time': times,
                 'band': bands,
                 'gain': gains,
                 'skynoise': skynoise,
                 'zp': zp,
                 'zpsys': zpsys})
    model = sncosmo.Model(source='salt2-extended')
    p = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': colour}
    model.set(z=z)
    lc = sncosmo.realize_lcs(obs, model, [p])[0]

    ston = (lc["flux"] / lc["fluxerr"]).max()

    filename = temp_dir + os.sep + "save_%d.npy" % seed

    if not os.path.exists(filename):
        model.set(t0=t0, x1=x1, c=colour, x0=x0)
        res, fitted_model = sncosmo.fit_lc(lc, model, ['t0', 'x0', 'x1', 'c'],
                                           guess_amplitude=False, guess_t0=False)
        fig = sncosmo.plot_lc(lc, model=fitted_model, errors=res.errors)
        fig.savefig(temp_dir + os.sep + "lc_%d.png" % seed, bbox_inches="tight", dpi=300)
        my_model = PerfectRedshift([lc], [z], t0, name="posterior%d" % seed)
        sampler = EnsembleSampler(temp_dir=temp_dir, num_burn=1000, num_steps=2000)
        c = ChainConsumer()
        my_model.fit(sampler, chain_consumer=c)
        c.plot_walks(filename=temp_dir + os.sep + "walk_%d.png" % seed)
        c.add_chain(np.random.multivariate_normal(res.parameters[1:], res.covariance, size=int(1e5)),
                    name="Gaussian", parameters=["$t_0$", "$x_0$", "$x_1$", "$c$"])
        figfilename = filename.replace(".npy", ".png")
        c.plot(filename=figfilename,
               truth={"$t_0$": t0, "$x_0$": x0, "$x_1$": x1, "$c$": colour})
        alpha = 0.14
        beta = 3.15

        c2 = ChainConsumer()
        means = []
        stds = []
        for i in range(len(c.chains)):
            chain = c.chains[i]
            apparent_interp = generate_and_return()
            x0s = chain[:, c.parameters[i].index("$x_0$")]
            x1s = chain[:, c.parameters[i].index("$x_1$")]
            cs = chain[:, c.parameters[i].index("$c$")]
            a = apparent_interp(x1s, cs, grid=False) - (0.4 * np.log10(x0s / 1e-5))
            a += alpha * x1s - beta * cs
            means.append(a.mean())
            stds.append(np.std(a))
            c2.add_chain(a, parameters=[r"$\mu+M$"], name=c.names[i])
        diffmu = np.diff(means)[0]
        diffstd = np.diff(stds)[0]
        np.save(filename, np.array([diffmu, diffstd]))

    else:
        vals = np.load(filename)
        diffmu = vals[0]
        diffstd = vals[1]

    return diffmu, diffstd, x1, colour, num_obs, ston, deltat

if __name__ == "__main__":
    temp_dir = os.path.dirname(__file__) + "/output/randoms"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    n = 200
    res = Parallel(n_jobs=1, max_nbytes="20M", verbose=100, batch_size=1)(delayed(random_obs)(
        temp_dir, i) for i in range(n))
    res = np.array(res)

    print(res.shape)

