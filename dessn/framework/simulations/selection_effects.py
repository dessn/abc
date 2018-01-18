import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, skewnorm
from scipy.optimize import curve_fit
import os
import inspect


def des_sel(cov_scale=1.0, shift=None, type="G10"):
    sn, mean, cov = get_selection_effects_cdf("snana_data/DES3YR_DES_BHMEFF_AM%s" % type)
    if shift is None:
        shift = np.array([0.0, 0.2, 0.0, 0.0])
    mean += shift
    cov *= cov_scale
    return sn, mean, cov


def lowz_sel(cov_scale=1.0, shift=None, type="G10"):
    sn, mean, cov = get_selection_effects_skewnorm("snana_data/DES3YR_LOWZ_BHMEFF_%s" % type)
    if shift is not None:
        mean += shift
    cov *= cov_scale
    return sn, mean, cov


def get_data(base):
    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)
    folder = dir_name + "/" + base
    supernovae_files = [folder + "/" + f for f in os.listdir(folder) if f.startswith("all")]
    supernovae_data = [np.load(f) for f in supernovae_files]
    supernovae = np.vstack(tuple(supernovae_data))
    passed = supernovae > 100
    mags = supernovae - 100 * passed.astype(np.int)
    return mags, passed


def get_ratio(base_folder, cut_mag=19.75):
    mB_all, passed = get_data(base_folder)
    print("Got data to compute selection function")
    mB_passed = mB_all[passed]

    # Bin data and get ratio
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


def get_selection_effects_cdf(dump_npy, plot=False, cut_mag=20):
    binc, ratio, ratio_error, ratio_smooth, ratio_smooth_error = get_ratio(dump_npy, cut_mag=cut_mag)

    def cdf(b, mean, sigma, alpha, n):
        model = (1 - norm.cdf(b, loc=mean, scale=sigma)) * n + 10 * alpha
        return model

    threshold = 0.02
    red_chi2 = 100
    adj = 0.0001
    adj = 1
    while np.abs(red_chi2 - 1) > threshold:
        if red_chi2 > 1:
            adj *= 1.01
        else:
            adj *= 0.98
        # ratio_error_adj = ratio_error + adj
        ratio_error_adj = 0.001 + ratio_error * adj
        # print(ratio_error_adj)
        result = curve_fit(cdf, binc, ratio, p0=np.array([23.0, 1.0, 0.0, 0.5]), sigma=ratio_error_adj)
        vals, cov, *_ = result
        chi2 = np.sum(((ratio - cdf(binc, *vals)) / ratio_error_adj) ** 2)
        red_chi2 = chi2 / (len(binc) - 3)

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
        # fig.savefig("../../../papers/methods/figures/%s.png" % name, bbox_inches="tight", transparent=True)
        fig.savefig("../../../papers/methods/figures/%s.pdf" % name, bbox_inches="tight", transparent=True)

    return False, vals, cov


def get_selection_effects_skewnorm(dump_npy, plot=False, cut_mag=10):
    binc, ratio, ratio_error, ratio_smooth, ratio_smooth_error = get_ratio(dump_npy, cut_mag=cut_mag)

    def sknorm(b, mean, sigma, alpha, n):

        model = skewnorm.pdf(b, alpha, loc=mean, scale=sigma) * n
        return model

    threshold = 0.1
    red_chi2 = np.inf
    adj = 0.001
    goal = 1
    while np.abs(red_chi2 - goal) > threshold:
        if red_chi2 > goal:
            adj *= 1.2
        else:
            adj *= 0.8
        ratio_error_adj = ratio_smooth_error + adj
        result = curve_fit(sknorm, binc, ratio_smooth, p0=np.array([15.0, 1.0, 0.0, 0.5]),
                           sigma=ratio_error_adj, bounds=([10, 0, -10, 0], [30, 10, 10, 5.0]))
        vals, cov, *_ = result
        chi2 = np.sum(((ratio - sknorm(binc, *vals)) / ratio_error_adj) ** 2)
        red_chi2 = chi2 / (len(binc) - 3)

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
        #plt.show()
        ax.text(0.98, 0.95, "Combined LowZ Sample", verticalalignment='top', horizontalalignment='right', transform=ax.transAxes)
        # fig.savefig("../../../papers/methods/figures/%s.png" % name, bbox_inches="tight", transparent=True)
        fig.savefig("../../../papers/methods/figures/%s.pdf" % name, bbox_inches="tight", transparent=True)

    return True, vals, cov


if __name__ == "__main__":
    get_selection_effects_skewnorm("snana_data/DES3YR_LOWZ_BHMEFF_G10", plot=True)
    # get_selection_effects_skewnorm("snana_data/DES3YR_LOWZ_BHMEFF_C11", plot=True)
    # get_selection_effects_cdf("snana_data/DES3YR_DES_BHMEFF_AMG10", plot=True)
    # get_selection_effects_cdf("snana_data/DES3YR_DES_BHMEFF_AMC11", plot=True)
    # get_selection_effects_cdf("snana_data/DES3YR_DES_BHMEFF_CD", plot=True, cut_mag=18)
