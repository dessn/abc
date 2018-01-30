import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from dessn.framework.simulations.snana import SNANASimulation


def investigate_color(sim_names):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    bins = np.linspace(-4, 4, 100)

    for name in sim_names:
        simulation = SNANASimulation(1000, name)
        data_all = simulation.get_passed_supernova(-1)
        zs = data_all["redshifts"]
        meanz = zs.mean()
        below = zs < meanz
        s_cs = data_all["sim_colours"]
        obs = data_all["obs_mBx1c"]
        sigma_cs = np.sqrt(data_all["obs_mBx1c_cov"][:, 2, 2])
        cs = np.array(obs[:, 2])
        diff = cs - s_cs

        ax.hist(diff[below] / sigma_cs[below], bins=bins, histtype='step', label=name.replace("DES3YR_DES_", "") + " z<%0.2f" % meanz, normed=True)
        ax.hist(diff[~below] / sigma_cs[~below], bins=bins, histtype='step', label=name.replace("DES3YR_DES_", "") + " z>%0.2f" % meanz, normed=True)

    ax.plot(bins, norm.pdf(bins))
    ax.legend()
    plt.show()


def fit_colour_error(names):
    cmaps = "viridis", "magma", "jet"
    for cmap, name in zip(cmaps, names):
        simulation = SNANASimulation(1000, name)
        data_all = simulation.get_passed_supernova(-1)
        zs = data_all["redshifts"]
        num_z_bins = 50
        z_bins = np.linspace(zs.min(), zs.max()*0.999, num_z_bins)
        indexes = np.digitize(zs, bins=z_bins) - 1
        s_cs = data_all["sim_colours"]
        obs = data_all["obs_mBx1c"]
        sigma_cs = np.sqrt(data_all["obs_mBx1c_cov"][:, 2, 2])
        cs = np.array(obs[:, 2])
        diff = cs - s_cs
        pull = diff / sigma_cs

        zs = 0.5 * (z_bins[1:] + z_bins[:-1])
        datasets = [diff[indexes == i] - sigma_cs[indexes == i] for i in range(num_z_bins - 1)]
        # datasets = [pull[indexes == i] for i in range(num_z_bins - 1)]
        ys = np.array([np.std(d) for d in datasets])
        # ys = np.array([np.std(d) - 0.75 for d in datasets])
        uncert = 0.001 + np.array([y/np.sqrt(2*len(d) - 2) for d, y in zip(datasets, ys)])

        def model(k):
            k0, k1 = k
            vals = k0 * (1 + zs * k1)
            diff = ((ys - vals) / uncert)**2
            return diff.sum()

        fit_val = minimize(model, np.array([0, 0.2]))
        xs = fit_val["x"]
        plt.plot(zs, xs[0] * (1 + xs[1] * zs))
        print(fit_val)
        plt.errorbar(zs, ys, yerr=uncert, fmt='.')
        #
        # def test_kappa(k0, k1):
        #     ratio = k0 + k1 * zs
        #     pull2 = pull / ratio
        #     return stds
        #
        # xx, yy = np.meshgrid(np.linspace(1, 2, 30), np.linspace(0, 1, 30), indexing='ij')
        # zz = test_kappa(xx, yy)
        # levels = np.linspace(0, 0.5, 30)
        # cc = plt.contour(xx, yy, zz, levels=levels, cmap=cmap)
        # plt.clabel(cc, inline=1)
    plt.show()


if __name__ == "__main__":
    # sim_names = ["DES3YR_DES_SAMTEST_MAGSMEAR", "DES3YR_DES_BULK_G10_SKEW", "DES3YR_DES_BULK_C11_SKEW"]
    # sim_names = ["DES3YR_DES_SAMTEST_MAGSMEAR"]
    sim_names = ["DES3YR_DES_BHMEFF_AMG10", "DES3YR_DES_BHMEFF_AMC11"]
    # sim_names = ["DES3YR_LOWZ_BHMEFF_G10", "DES3YR_LOWZ_BHMEFF_C11"]
    # investigate_color(sim_names)
    fit_colour_error(sim_names)
