"""

This file is used to generate supernova light curves, perform
some simple cosmology fits on them without regarding selection
effects, and then see whether the differences between the
summary statistics fitting software cause any shift in output
cosmological parameters.

"""
from scipy.interpolate import RegularGridInterpolator

from dessn.investigations.gaussianity.simple_cosmology_fitter import SimpleCosmologyFitter
from dessn.investigations.gaussianity.des_sky import get_result, realise_light_curve
from dessn.framework.samplers.ensemble import EnsembleSampler
import os
from joblib import Parallel, delayed
import numpy as np
import logging
from astropy.cosmology import WMAP9, FlatwCDM


def get_zp_and_name(shallow):
    if shallow:
        return [32.46, 32.28, 32.55, 33.12], "shallow"
    else:
        return [34.24, 34.85, 34.94, 35.42], "deep"


def get_supernova_data(n=600, ston_thresh=5, shallow=True):
    zp, name = get_zp_and_name(shallow)
    temp_dir = os.path.dirname(__file__) + "/output/supernova_data_%s" % name
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    ress = Parallel(n_jobs=4, max_nbytes="20M", batch_size=5)(delayed(get_result)(
        temp_dir, zp, i, 0.1, False, False) for i in range(n))
    res = np.array([r for r in ress if r is not None])
    ston = res[:, 6]
    res = res[ston > ston_thresh, :]
    z = res[:, 1]
    # res = res[z < 0.7, :]
    # [seed, z, t0, x0, x1, c, ston] + mus + stds)
    #            zs, mu_mcmc, mu_minuit, std_mcmc, std_minuit
    return res[:, 1], res[:, 7], res[:, 8], res[:, 9], res[:, 10]


def get_bias(omega_m, w0, mabs, n=800, ston_thresh=5, shallow=True):
    zp, name = get_zp_and_name(shallow)
    cosmology = FlatwCDM(70, omega_m, w0=w0)
    ress = Parallel(n_jobs=4, max_nbytes="20M", batch_size=100)(delayed(realise_light_curve)(
        None, zp, i, 0.1, cosmology, mabs) for i in range(n))
    res = np.array([r for r in ress if r is not None])
    ston = res[:, 5]
    mask = (ston > ston_thresh).sum()
    return 1.0 * mask / n


def get_bias_matrix(filename, shallow=True):
    omega_ms = np.linspace(0.1, 0.6, 7)
    ws = np.linspace(-2, -0.5, 7)
    mabss = np.linspace(-20, -19, 7)

    if os.path.exists(filename):
        bs = np.load(filename)
    else:
        bs = np.zeros((omega_ms.size, ws.size, mabss.size)) - 1
    if np.any(bs < 0.0):
        print("Starting bias calculation")
        for i, m in enumerate(omega_ms):
            for j, w in enumerate(ws):
                for k, a in enumerate(mabss):
                    if bs[i, j, k] < 0:
                        bs[i, j, k] = get_bias(m, w, a, shallow=shallow)
                    else:
                        print(bs[i, j, k])
                print("...%0.2f" % (1.0 * (j + 1) / len(ws)))
                np.save(filename, bs)
            print("%0.2f" % ((i + 1) / omega_ms.size))
        np.save(filename, bs)

    return RegularGridInterpolator((omega_ms, ws, mabss), bs, bounds_error=False, fill_value=0.0)


def plot_cosmology(zs, mu_mcmc, mu_minuit, std_mcmc, std_minuit):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    # ax.axhline(0, color='k', ls="--")
    zsort = sorted(zs)
    distmod = WMAP9.distmod(zsort).value - 19.3
    distmod2 = FlatwCDM(70, 0.1, -0.8).distmod(zsort).value - 19.3
    distmod3 = FlatwCDM(70, 0.25, -2).distmod(zsort).value - 19.5
    ax.errorbar(zs, mu_minuit, yerr=np.sqrt(std_minuit*std_minuit + 0.1*0.1), ms=4, fmt='o', label=r"minuit", color="r", alpha=0.1)
    # ax.errorbar(zs, mu_mcmc, yerr=np.sqrt(std_mcmc*std_mcmc + 0.05*0.05), fmt='o', ms=4, label=r"mcmc", color="b", alpha=0.1)
    ax.plot(zsort, distmod, 'k')
    ax.plot(zsort, distmod2, 'g--')
    ax.plot(zsort, distmod3, 'r:')
    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$\mu(\mathcal{C}) - \mu_{{\rm obs}}$")
    ax.legend(loc=2)
    fig.savefig("output/obs_cosmology.png", bbox_inches="tight", dpi=300, transparent=True)

if __name__ == "__main__":
    bias_file = os.path.dirname(__file__) + "/output/cosmology/bias_shallow.npy"
    temp_dir2 = os.path.dirname(__file__) + "/output/cosmology"
    if not os.path.exists(temp_dir2):
        os.makedirs(temp_dir2)
    logging.basicConfig(level=logging.DEBUG)

    interp = get_bias_matrix(bias_file)
    zs, mu_mcmc, mu_minuit, std_mcmc, std_minuit = get_supernova_data()

    plot_cosmology(zs, mu_mcmc, mu_minuit, std_mcmc, std_minuit)
    fitter_mcmc = SimpleCosmologyFitter("mcmc", zs, mu_mcmc, std_mcmc, interp)
    fitter_minuit = SimpleCosmologyFitter("minuit", zs, mu_minuit, std_minuit, interp)

    sampler = EnsembleSampler(temp_dir=temp_dir2, save_interval=60, num_steps=5000, num_burn=1000)
    c = fitter_mcmc.fit(sampler=sampler)
    c = fitter_minuit.fit(sampler=sampler, chain_consumer=c)
    c.names = ["emcee", "minuit"]
    #c.plot_walks(filename="output/walks.png")
    c.plot(filename="output/comparison_shallow.png", parameters=2, figsize=(5.5, 5.5))
    print(c.get_latex_table())