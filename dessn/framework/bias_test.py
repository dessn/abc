import numpy as np
from simulations.snana import SNANASimulationGauss0p3, SNANASimulationIdeal0p3, SNANASimulationIdealNoBias0p3
import matplotlib.pyplot as plt
from scipy.stats import norm


def inspect_bias(simulation):
    data_all = simulation.get_all_supernova(-1)

    mean, sigma, alpha, normalisation = simulation.get_approximate_correction()
    mv, mp = 30, 20
    nbin = [50, 40]
    passed = data_all["passed"]
    zs = data_all["redshifts"]
    mbs = data_all["sim_apparents"]
    cs = data_all["sim_colors"]
    x1s = data_all["sim_stretches"]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 5))
    axes = axes.flatten()

    hist_all, binx, biny = np.histogram2d(mbs, zs, bins=nbin)
    hist_passed, _, _ = np.histogram2d(mbs[passed], zs[passed], bins=[binx, biny])
    mask = (hist_all.T < mv) | (hist_passed.T < mp)
    binxc = 0.5 * (binx[:-1] + binx[1:])
    correction = 1 / ((1 - norm.cdf(binxc, mean, sigma)) + 0.015)
    ratio = hist_passed / (1 + hist_all)
    ratio = ratio.T
    ratio[mask] = np.nan
    ratio2 = ratio * correction
    hist_passed = hist_passed.T
    hist_passed[mask] = np.nan
    axes[0].imshow(ratio, interpolation="nearest", origin="lower",
                   extent=[binx[0], binx[-1], biny[0], biny[-1]], aspect="auto", cmap="viridis")
    axes[3].imshow(ratio2, interpolation="nearest", origin="lower",
                   extent=[binx[0], binx[-1], biny[0], biny[-1]], aspect="auto", cmap="viridis")
    axes[6].imshow(hist_passed, interpolation="nearest", origin="lower",
                   extent=[binx[0], binx[-1], biny[0], biny[-1]], aspect="auto", cmap="viridis")
    hist_all, binx, biny = np.histogram2d(mbs, cs, bins=nbin)
    hist_passed, _, _ = np.histogram2d(mbs[passed], cs[passed], bins=[binx, biny])
    mask = (hist_all.T < mv) | (hist_passed.T < mp)
    binxc = 0.5 * (binx[:-1] + binx[1:])
    correction = 1 / ((1 - norm.cdf(binxc, mean, sigma)) + 0.015)
    ratio = hist_passed / (1 + hist_all)
    ratio = ratio.T
    ratio[mask] = np.nan
    ratio2 = ratio * correction
    hist_passed = hist_passed.T
    hist_passed[mask] = np.nan
    axes[1].imshow(ratio, interpolation="nearest", origin="lower",
                   extent=[binx[0], binx[-1], biny[0], biny[-1]], aspect="auto", cmap="viridis")
    axes[4].imshow(ratio2, interpolation="nearest", origin="lower",
                   extent=[binx[0], binx[-1], biny[0], biny[-1]], aspect="auto", cmap="viridis")
    axes[7].imshow(hist_passed, interpolation="nearest", origin="lower",
                   extent=[binx[0], binx[-1], biny[0], biny[-1]], aspect="auto", cmap="viridis")

    hist_all, binx, biny = np.histogram2d(mbs, x1s, bins=nbin)
    hist_passed, _, _ = np.histogram2d(mbs[passed], x1s[passed], bins=[binx, biny])
    mask = (hist_all.T < mv) | (hist_passed.T < mp)
    binxc = 0.5 * (binx[:-1] + binx[1:])
    correction = 1 / ((1 - norm.cdf(binxc, mean, sigma)) + 0.015)
    ratio = hist_passed / (1 + hist_all)
    ratio = ratio.T
    ratio[mask] = np.nan
    ratio2 = ratio * correction
    hist_passed = hist_passed.T
    hist_passed[mask] = np.nan
    axes[2].imshow(ratio, interpolation="nearest", origin="lower",
                   extent=[binx[0], binx[-1], biny[0], biny[-1]], aspect="auto", cmap="viridis")
    axes[5].imshow(ratio2, interpolation="nearest", origin="lower",
                   extent=[binx[0], binx[-1], biny[0], biny[-1]], aspect="auto", cmap="viridis")
    axes[8].imshow(hist_passed, interpolation="nearest", origin="lower",
                   extent=[binx[0], binx[-1], biny[0], biny[-1]], aspect="auto", cmap="viridis")
    print(binx, biny)

    plt.savefig(simulation.__class__.__name__ + "BiasCorrection.png")
    plt.show()

if __name__ == "__main__":
    # inspect_bias(SNANASimulationIdeal0p3(-1))
    inspect_bias(SNANASimulationIdealNoBias0p3(-1))
