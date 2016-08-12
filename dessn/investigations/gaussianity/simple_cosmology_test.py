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
import matplotlib.pyplot as plt


def get_zp_and_name(shallow):
    if shallow:
        return [32.46, 32.28, 32.55, 33.12], "shallow"
    else:
        return [34.24, 34.85, 34.94, 35.42], "deep"


def get_supernova_data(n=1500, ston_thresh=5, shallow=True):
    zp, name = get_zp_and_name(shallow)
    temp_dir = os.path.dirname(__file__) + "/output/supernova_data_%s" % name
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    ress = Parallel(n_jobs=4, max_nbytes="20M", batch_size=5)(delayed(get_result)(
        temp_dir, zp, i, 0.1, False, False) for i in range(n))
    res = np.array([r for r in ress if r is not None])
    ston = res[:, 6]
    res = res[ston > ston_thresh, :]

    diff = np.abs(res[:, 7] - (WMAP9.distmod(res[:, 1]).value - 19.3))
    print(diff)
    res = res[(diff < 4), :]
    # [seed, z, t0, x0, x1, c, ston] + mus + stds)
    #            zs, mu_mcmc, mu_minuit, std_mcmc, std_minuit
    print("Supernova data", res.shape)
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


def plot_cosmology(zs, mu_mcmc, mu_minuit, std_mcmc, std_minuit, n):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.ticker import MaxNLocator
    fig = plt.figure(figsize=(4.5, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.0, wspace=0.0)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    axes = [ax0, ax1]
    zsort = sorted(zs)
    distmod = WMAP9.distmod(zsort).value - 19.3
    distmod2 = WMAP9.distmod(zs).value - 19.3
    ms = 2
    alpha = 0.4
    axes[0].errorbar(zs, mu_minuit, yerr=np.sqrt(std_minuit*std_minuit + 0.1*0.1),
                     ms=ms, fmt='o', label=r"minuit", color="r", alpha=alpha)
    axes[0].errorbar(zs, mu_mcmc, yerr=np.sqrt(std_mcmc*std_mcmc + 0.1*0.1), fmt='o',
                     ms=ms, label=r"mcmc", color="b", alpha=alpha)
    axes[1].errorbar(zs, mu_minuit - distmod2, yerr=np.sqrt(std_minuit*std_minuit + 0.1*0.1),
                     ms=ms, fmt='o', label=r"minuit", color="r", alpha=alpha)
    axes[1].errorbar(zs, mu_mcmc - distmod2, yerr=np.sqrt(std_mcmc*std_mcmc + 0.1*0.1), fmt='o',
                     ms=ms, label=r"mcmc", color="b", alpha=alpha)
    axes[0].plot(zsort, distmod, 'k')
    axes[1].axhline(0, color='k')
    axes[1].set_xlabel("$z$")
    axes[0].set_ylabel(r"$\mu$")
    axes[1].set_ylabel(r"$\mu_{\rm obs} - \mu(\mathcal{C})$")
    axes[0].legend(loc=2, frameon=False)
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.yaxis.set_major_locator(MaxNLocator(7, prune="lower"))
    ax1.yaxis.set_major_locator(MaxNLocator(3))

    fig.savefig("output/obs_cosmology_%s.png" % n, bbox_inches="tight", dpi=300, transparent=True)
    fig.savefig("output/obs_cosmology_%s.pdf" % n, bbox_inches="tight", dpi=300, transparent=True)

if __name__ == "__main__":
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    for n in ["deep", "shallow"]:
        is_shallow = n == "shallow"
        bias_file = os.path.dirname(__file__) + "/output/cosmology/bias_%s.npy" % n
        temp_dir2 = os.path.dirname(__file__) + "/output/cosmology_%s" % n
        if not os.path.exists(temp_dir2):
            os.makedirs(temp_dir2)
        logging.basicConfig(level=logging.DEBUG)

        zs, mu_mcmc, mu_minuit, std_mcmc, std_minuit = get_supernova_data(shallow=is_shallow)

        plot_cosmology(zs, mu_mcmc, mu_minuit, std_mcmc, std_minuit, n)
        fitter_mcmc = SimpleCosmologyFitter("mcmc", zs, mu_mcmc, std_mcmc)
        fitter_minuit = SimpleCosmologyFitter("minuit", zs, mu_minuit, std_minuit)

        sampler = EnsembleSampler(temp_dir=temp_dir2, save_interval=60, num_steps=25000, num_burn=1000)
        c = fitter_mcmc.fit(sampler=sampler)
        c = fitter_minuit.fit(sampler=sampler, chain_consumer=c)
        c.names = ["emcee", "minuit"]
        c.plot(filename="output/comparison_%s.png" % n, parameters=2, figsize=(5.5, 5.5))
        c.plot(filename="output/comparison_%s.pdf" % n, parameters=2, figsize=(5.5, 5.5))
        print(c.get_latex_table())