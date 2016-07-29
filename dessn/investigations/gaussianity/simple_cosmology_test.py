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

if __name__ == "__main__":
    temp_dir = os.path.dirname(__file__) + "/output/dessky"
    temp_dir2 = os.path.dirname(__file__) + "/output/cosmology"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if not os.path.exists(temp_dir2):
        os.makedirs(temp_dir2)
    logging.basicConfig(level=logging.DEBUG)

    n = 450
    res = Parallel(n_jobs=4, max_nbytes="20M", batch_size=5)(delayed(get_result)(
        temp_dir, i) for i in range(n))
    res = np.array([r for r in res if r is not None])
    s = res[:, 6]
    res = res[(s > 8), :]  # Cut low signal to noise

    seeds = res[:, 0]
    z = res[:, 1]
    s = res[:, 6] / 100
    skews = res[:, -1]
    mu_minuit = res[:, 9]
    mu_mcmc = res[:, 10]
    std_minuit = res[:, 14]
    std_mcmc = res[:, 15]

    import matplotlib.pyplot as plt
    plt.scatter(z, mu_mcmc)
    plt.show()
    exit()

    fitter_mcmc = SimpleCosmologyFitter("mcmc fit", z, mu_mcmc, std_mcmc)
    fitter_minuit = SimpleCosmologyFitter("minuit fit", z, mu_minuit, std_minuit)

    sampler = EnsembleSampler(temp_dir=temp_dir2, save_interval=60,
                              num_steps=400, num_burn=0)

    c = fitter_mcmc.fit(sampler=sampler)
    # c = fitter_minuit.fit(sampler=sampler, chain_consumer=c)
    c.plot_walks(filename="output/walks.png")
    c.plot(filename="output/comparison.png")