from dessn.investigations.gaussianity.model import PerfectRedshift
import sncosmo
import numpy as np
from astropy.table import Table
from dessn.chain.chain import ChainConsumer
from dessn.framework.samplers.ensemble import EnsembleSampler
import os


if __name__ == "__main__":
    np.random.seed(0)
    x1 = np.random.normal()
    c = np.random.normal(scale=0.1)
    x0 = np.random.normal(loc=1e-5, scale=1e-6)
    t0 = np.random.uniform(low=1000, high=2000)
    z = np.random.uniform(low=0.1, high=0.7)

    num_obs = 30
    ts = np.linspace(t0 - 60, t0 + 60, num_obs)

    times = np.array([[t, t + 0.1, t + 0.2] for t in ts]).flatten()
    bands = [b for t in ts for b in ['desg', 'desr', 'desi']]
    gains = np.ones(times.shape)
    skynoise = 20 * np.ones(times.shape)
    zp = 30 * np.ones(times.shape)
    zpsys = ['ab'] * times.size

    obs = Table({'time': times,
                 'band': bands,
                 'gain': gains,
                 'skynoise': skynoise,
                 'zp': zp,
                 'zpsys': zpsys})

    model = sncosmo.Model(source='salt2')
    p = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}
    model.set(z=z)
    lcs = sncosmo.realize_lcs(obs, model, [p])
    res, fitted_model = sncosmo.fit_lc(lcs[0], model, ['t0', 'x0', 'x1', 'c'])

    dir_name = os.path.dirname(__file__)
    temp_dir = dir_name + os.sep + "output"
    surface = temp_dir + os.sep + "surfaces_simple.png"
    c = ChainConsumer()
    my_model = PerfectRedshift(lcs, [z])
    sampler = EnsembleSampler(temp_dir=temp_dir, num_steps=10000)
    my_model.fit(sampler, chain_consumer=c)
    c.add_chain(np.random.multivariate_normal(res.parameters[1:], res.covariance, size=int(1e6)),
                name="Gaussian", parameters=["$t_0$", "$x_0$", "$x_1$", "$c$"])

    c.configure_contour(contourf=True, contourf_alpha=0.2, sigmas=[0.0, 0.5, 1.0, 2.0, 3.0])
    c.configure_bar(shade=True)
    c.plot(filename=surface, figsize=(7, 7))
    fig = sncosmo.plot_lc(lcs[0], model=fitted_model, errors=res.errors)
    fig.savefig(temp_dir + os.sep + "lc_simple.png", bbox_inches="tight", dpi=300)
