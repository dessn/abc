from dessn.investigations.gaussianity.model import PerfectRedshift
from dessn.investigations.gaussianity.apparent_mags import generate_and_return
import sncosmo
import numpy as np
from astropy.table import Table
from dessn.chain.chain import ChainConsumer
from dessn.framework.samplers.ensemble import EnsembleSampler
import os


if __name__ == "__main__":
    np.random.seed(0)
    x1 = np.random.normal()
    colour = np.random.normal(scale=0.1)
    x0 = np.random.normal(loc=1e-5, scale=1e-6)
    t0 = np.random.uniform(low=1000, high=2000)
    z = np.random.uniform(low=0.1, high=0.7)

    num_obs = 30
    ts = np.linspace(t0 - 60, t0 + 60, num_obs)

    times = np.array([[t, t + 0.1, t + 0.2, t + 0.3] for t in ts]).flatten()
    bands = [b for t in ts for b in ['desg', 'desr', 'desi', 'desz']]
    gains = np.ones(times.shape)
    skynoise = 80 * np.ones(times.shape)
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
    lcs = sncosmo.realize_lcs(obs, model, [p])
    res, fitted_model = sncosmo.fit_lc(lcs[0], model, ['t0', 'x0', 'x1', 'c'])

    dir_name = os.path.dirname(__file__)
    temp_dir = dir_name + os.sep + "output"
    surface = temp_dir + os.sep + "surfaces_simple.png"
    mu_simple = temp_dir + os.sep + "mu_simple.png"
    mcmc_chain = temp_dir + os.sep + "mcmc_simple.npy"
    c = ChainConsumer()
    my_model = PerfectRedshift(lcs, [z], t0, name="My posterior")
    sampler = EnsembleSampler(temp_dir=temp_dir, num_steps=10000)
    my_model.fit(sampler, chain_consumer=c)
    c.add_chain(np.random.multivariate_normal(res.parameters[1:], res.covariance, size=int(1e6)),
                name="Summary Stats", parameters=["$t_0$", "$x_0$", "$x_1$", "$c$"])

    if False:
        if not os.path.exists(mcmc_chain):
            res2, fitted_model2 = sncosmo.mcmc_lc(lcs[0], model, ['t0', 'x0', 'x1', 'c'], nwalkers=20,
                                                  nburn=500, nsamples=4000)
            mcchain = res2.samples
            np.save(mcmc_chain, mcchain)
        else:
            mcchain = np.load(mcmc_chain)
        c.add_chain(mcchain, name="sncosmo mcmc", parameters=["$t_0$", "$x_0$", "$x_1$", "$c$"])

    c.configure_contour(contourf=True, contourf_alpha=0.2, sigmas=[0.0, 0.5, 1.0, 2.0, 3.0])
    c.configure_bar(shade=True)
    c.plot(filename=surface, figsize=(7, 7))
    fig = sncosmo.plot_lc(lcs[0], model=fitted_model, errors=res.errors)
    fig.savefig(temp_dir + os.sep + "lc_simple.png", bbox_inches="tight", dpi=300)

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
    print(means, stds, np.diff(means), np.diff(stds))

    actual = apparent_interp(x1, colour, grid=False) - (0.4 * np.log10(x0 / 1e-5))
    actual += alpha * x1 - beta * colour
    c2.plot(filename=mu_simple, figsize=(7, 4), truth=[actual], legend=True)

