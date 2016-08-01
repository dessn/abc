"""

This file is used to generate supernova light curves, perform
some simple cosmology fits on them without regarding selection
effects, and then see whether the differences between the
summary statistics fitting software cause any shift in output
cosmological parameters.

"""

from dessn.investigations.gaussianity.simple_cosmology_fitter import SimpleCosmologyFitter
from dessn.investigations.gaussianity.des_sky import get_result
from dessn.framework.samplers.ensemble import EnsembleSampler
import os
from joblib import Parallel, delayed
import numpy as np
import logging
from astropy.cosmology import WMAP9


if __name__ == "__main__":
    temp_dir = os.path.dirname(__file__) + "/output/example"
    temp_dir2 = os.path.dirname(__file__) + "/output/cosmology"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if not os.path.exists(temp_dir2):
        os.makedirs(temp_dir2)
    logging.basicConfig(level=logging.DEBUG)

    n = 500
    ress = Parallel(n_jobs=4, max_nbytes="20M", batch_size=5)(delayed(get_result)(
        temp_dir, [32.46, 32.28, 32.55, 33.12], i, 0.01) for i in range(n))

    for cut in [5, 8]:

        res = np.array([r for r in ress if r is not None])

        s = res[:, 6]
        res = res[(s > 8), :]  # Cut low signal to noise
        zs = res[:, 1]
        res = res[(zs < 1.7), :]

        seeds = res[:, 0]
        zs = res[:, 1]
        s = res[:, 6] / 100
        skews = res[:, -1]
        mu_minuit = res[:, 9]
        mu_mcmc = res[:, 10]
        std_minuit = res[:, 14]
        std_mcmc = res[:, 15]

        fitter_mcmc = SimpleCosmologyFitter("mcmc fit %d" % cut, zs, mu_mcmc, std_mcmc)
        fitter_minuit = SimpleCosmologyFitter("minuit fit %d" % cut, zs, mu_minuit, std_minuit)

        distmod = WMAP9.distmod(zs).value - 19.3
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.axhline(0, color='k', ls="--")
        ax.errorbar(zs, distmod - mu_minuit, yerr=std_minuit, ms=4, fmt='o', label=r"minuit", color="r")
        ax.errorbar(zs, distmod - mu_mcmc, yerr=std_mcmc, fmt='o', ms=4, label=r"mcmc", color="b")
        ax.set_xlabel("$z$")
        ax.set_ylabel(r"$\mu(\mathcal{C}) - \mu_{{\rm obs}}$")
        ax.legend(loc=2)
        fig.savefig("output/obs_cosmology.png", bbox_inches="tight", dpi=300, transparent=True)

        sampler = EnsembleSampler(temp_dir=temp_dir2, save_interval=60,
                                  num_steps=80000, num_burn=2000)
        from chainconsumer import ChainConsumer
        c = ChainConsumer()
        c = fitter_mcmc.fit(sampler=sampler, chain_consumer=c)
        c = fitter_minuit.fit(sampler=sampler, chain_consumer=c)
        c.names = ["emcee", "minuit"]
        c.plot(filename="output/comparison%d.png" % cut, parameters=2, figsize=(5.5, 5.5))
        print(c.get_latex_table())