import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, skewnorm
from scipy.optimize import curve_fit, minimize
import os
import inspect
import logging


def des_sel(cov_scale=1.0, shift=None, type="G10", kappa=0, zlim=None, version=None):
    if type is None:
        name = "snana_data/DES3YR_DES_BHMEFF_CD"
    else:
        name = "snana_data/DES3YR_DES_BHMEFF_AM%s" % type
    if version is not None:
        logging.info("Using version %s" % version)
        name += "_%s" % version
    sn, mean, cov, _ = get_selection_effects_cdf(name, kappa=kappa, zlim=zlim)
    if shift is None:
        shift = np.array([0.0, 0, 0.0, 0.0])
    mean += shift
    logging.info("Getting DES selection, shift of %s" % shift)
    cov *= cov_scale
    return sn, mean, cov, kappa


def lowz_sel(cov_scale=1.0, shift=None, type="G10", kappa=0, zlim=None, version=None):
    if type is None:
        type = "G10"
    name = "snana_data/DES3YR_LOWZ_BHMEFF_%s" % type
    if version is not None:
        logging.info("Using version %s" % version)
        name += "_%s" % version
    sn, mean, cov, _ = get_selection_effects_skewnorm(name, kappa=kappa, zlim=zlim)
    if shift is None:
        shift = np.array([0.0, 0.0, 0.0, 0.0])
    mean += shift
    logging.info("Getting LOWZ selection, shift of %s" % shift)
    cov *= cov_scale
    return sn, mean, cov, kappa


def get_data(base, zlim=None, maxc=None, minc=None):
    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)
    folder = dir_name + "/" + base
    supernovae_files = [folder + "/" + f for f in os.listdir(folder) if f.startswith("all")]
    supernovae_data = [np.load(f) for f in supernovae_files]
    supernovae = np.vstack(tuple(supernovae_data))
    passed = supernovae[:, 0] > 100
    mags = supernovae[:, 0] - 100 * passed.astype(np.int)
    zs = supernovae[:, 1]
    if zlim is not None:
        mask = zs < zlim
        mags = mags[mask]
        passed = passed[mask]
    return mags, passed


def get_ratio(base_folder, cut_mag=19.75, delta=0, zlim=None, maxc=None, minc=None):
    mB_all, passed = get_data(base_folder, zlim=zlim, maxc=maxc, minc=minc)
    mB_passed = mB_all[passed]

    # Bin data and get ratio
    # import matplotlib.pyplot as plt
    # plt.hist(mB_passed, 100)
    # plt.show()
    # exit()
    hist_all, bins = np.histogram(mB_all, bins=100)
    hist_passed, _ = np.histogram(mB_passed, bins=bins)
    hist_passed_err = np.sqrt(hist_passed)

    binc = 0.5 * (bins[:-1] + bins[1:])
    keep = binc > cut_mag

    binc = binc[keep]
    hist_all[hist_all == 0] = 1
    ratio = hist_passed[keep] / hist_all[keep]
    ratio_error = ratio * (hist_passed_err[keep] / (hist_passed[keep] + 0.01))
    ratio_smooth = gaussian_filter1d(ratio, 2)
    ratio_smooth_error = gaussian_filter1d(ratio_error, 2)

    return binc, ratio, ratio_error, ratio_smooth, ratio_smooth_error


def get_selection_effects_cdf(dump_npy, plot=False, cut_mag=19, kappa=0, zlim=None):
    binc, ratio, ratio_error, ratio_smooth, ratio_smooth_error = get_ratio(dump_npy, cut_mag=cut_mag, delta=kappa, zlim=zlim)
    # print(binc, ratio)
    def cdf(b, mean, sigma, alpha, n):
        model = (1 - norm.cdf(b, loc=mean, scale=sigma)) * n + 10 * alpha
        return model

    threshold = 0.02
    red_chi2 = 100
    adj = 0.0001
    r2 = None
    while np.abs(red_chi2 - 1) > threshold:
        if red_chi2 > 1:
            adj *= 1.01
        else:
            adj *= 0.98
        ratio_error_adj = np.sqrt(ratio_error**2 + adj**2)
        # ratio_error_adj = 0.001 + ratio_error * adj
        # print(ratio_error_adj)
        result = curve_fit(cdf, binc, ratio, p0=np.array([23.0, 1.0, 0.0, 0.5]), sigma=ratio_error_adj)
        vals, cov, *_ = result
        chi2 = np.sum(((ratio - cdf(binc, *vals)) / ratio_error_adj) ** 2)
        red_chi2 = chi2 / (len(binc) - 3)
        if r2 is None:
            r2 = red_chi2

    if plot:
        import matplotlib.pyplot as plt
        from matplotlib import rc
        rc('text', usetex=True)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.errorbar(binc, ratio, yerr=ratio_error_adj, ls="none", fmt='o', ms=3, label="Simulation")
        #ax.errorbar(binc, ratio_smooth, yerr=ratio_smooth_error, label="Bin(err) (smoothed)")

        mbs = np.linspace(binc[0], binc[-1], 1000)
        cdf_eval = cdf(mbs, *vals)

        for i in range(50):
            rands = np.random.multivariate_normal([0, 0, 0, 0], cov=cov)
            vals2 = vals + rands
            cdf_eval2 = cdf(mbs, *vals2)
            plt.plot(mbs, cdf_eval2, c='k', alpha=0.05)
        ax.set_ylabel("Probability")
        ax.set_xlabel(r"$m_B$")
        ax.plot(mbs, cdf_eval, c='k', label="Fitted efficiency")
        ax.legend(frameon=False, loc=3)
        ax.set_xlim(20, 24)
        fig.tight_layout()
        name = os.path.basename(dump_npy)

        ax.text(0.98, 0.95, "DES 3YR Spectroscopically Confirmed", verticalalignment='top', horizontalalignment='right', transform=ax.transAxes)
        fig.savefig("../../../papers/methods/figures/%s.pdf" % name, bbox_inches="tight", transparent=True)
        plt.show()

    print(vals, cov, r2)
    return False, vals, cov, r2


def get_selection_effects_skewnorm(dump_npy, plot=False, cut_mag=10, kappa=0, zlim=None):
    binc, ratio, ratio_error, ratio_smooth, ratio_smooth_error = get_ratio(dump_npy, delta=kappa, cut_mag=cut_mag, zlim=zlim)

    def sknorm(b, mean, sigma, alpha, n):

        model = skewnorm.pdf(b, alpha, loc=mean, scale=sigma) * n
        return model

    threshold = 0.1
    red_chi2 = np.inf
    adj = 0.0001
    goal = 1
    r2 = None
    while np.abs(red_chi2 - goal) > threshold:
        if red_chi2 > goal:
            adj *= 1.2
        else:
            adj *= 0.8
        # ratio_error_adj = ratio_smooth_error + adj
        ratio_error_adj = np.sqrt(ratio_smooth_error**2 + adj**2)
        result = curve_fit(sknorm, binc, ratio_smooth, p0=np.array([15.0, 1.0, 0.0, 0.5]),
                           sigma=ratio_error_adj, bounds=([10, 0, -10, 0], [30, 10, 10, 5.0]))
        vals, cov, *_ = result
        chi2 = np.sum(((ratio - sknorm(binc, *vals)) / ratio_error_adj) ** 2)
        red_chi2 = chi2 / (len(binc) - 3)
        if r2 is None:
            r2 = red_chi2

    if plot:
        import matplotlib.pyplot as plt
        from matplotlib import rc
        rc('text', usetex=True)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.errorbar(binc, ratio, yerr=ratio_error_adj, ls="none", fmt='o', ms=3, label="Simulation")
        # ax.errorbar(binc, ratio_smooth, yerr=ratio_smooth_error)

        mbs = np.linspace(binc[0], binc[-1], 1000)
        cdf_eval = sknorm(mbs, *vals)

        for i in range(50):
            rands = np.random.multivariate_normal([0, 0, 0, 0], cov=cov)
            vals2 = vals + rands
            cdf_eval2 = sknorm(mbs, *vals2)
            plt.plot(mbs, cdf_eval2, c='k', alpha=0.05)

        ax.set_ylabel("Probability")
        ax.set_xlabel(r"$m_B$")
        ax.plot(mbs, cdf_eval, c='k', label="Fitted efficiency")
        ax.legend(frameon=False, loc=2)
        ax.set_xlim(13, 17)
        ax.set_ylim(0, 0.4)
        fig.tight_layout()
        name = os.path.basename(dump_npy)
        plt.show()
        ax.text(0.98, 0.95, "Low-$z$ Sample", verticalalignment='top', horizontalalignment='right', transform=ax.transAxes)
        fig.savefig("../../../papers/methods/figures/%s.png" % name, bbox_inches="tight", transparent=True)
        fig.savefig("../../../papers/methods/figures/%s.pdf" % name, bbox_inches="tight", transparent=True)

    return True, vals, cov, r2


def test_colour_contribution():
    ds = []
    a1 = []
    a2 = []
    for delta in np.linspace(-2.5, -4, 40):
        sn, mean, cov, adj = get_selection_effects_cdf("snana_data/DES3YR_DES_BHMEFF_AMG10", kappa=delta)
        _, _, _, adj2 = get_selection_effects_cdf("snana_data/DES3YR_DES_BHMEFF_AMC11", kappa=delta)
        # sn, mean, cov, adj = get_selection_effects_skewnorm("snana_data/DES3YR_LOWZ_BHMEFF_G10", kappa=delta)
        # _, _, _, adj2 = get_selection_effects_skewnorm("snana_data/DES3YR_LOWZ_BHMEFF_C11", kappa=delta)
        print("%5.2f %5.2f %5.2f" % (delta, adj, adj2))
        ds.append(delta)
        a1.append(adj)
        a2.append(adj2)
        # print(mean)
        # print(np.sqrt(np.diag(cov)))
    import matplotlib.pyplot as plt
    plt.plot(ds, a1)
    plt.plot(ds, a2)
    plt.show()

if __name__ == "__main__":
    get_selection_effects_skewnorm("snana_data/DES3YR_LOWZ_BHMEFF_G10", plot=True)
    # get_selection_effects_skewnorm("snana_data/DES3YR_LOWZ_BHMEFF_C11", plot=True)
    # test_colour_contribution()
    get_selection_effects_cdf("snana_data/DES3YR_DES_BHMEFF_AMG10", plot=True)
    print("---")
    # get_selection_effects_cdf("snana_data/DES3YR_DES_BHMEFF_AMC11", plot=True)
    # print("---")
    # get_selection_effects_cdf("snana_data/DES3YR_DES_BHMEFF_CD", plot=True, cut_mag=19)

    # _, mean, cov = get_selection_effects_cdf("snana_data/DES3YR_DES_BHMEFF_AMG10")
    # print(mean, np.sqrt(np.diag(cov)))
    # _, mean, cov = get_selection_effects_cdf("snana_data/DES3YR_DES_BHMEFF_AMC11")
    # print(mean, np.sqrt(np.diag(cov)))
