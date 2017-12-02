import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm
from scipy.optimize import curve_fit
import os
import inspect


def des_sel(cov_scale=1.0):
    sn, mean, cov = get_selection_effects_cdf("snana_data/DES3Y_DES_BHMEFF_AMG10/all.npy")
    cov *= cov_scale
    return sn, mean, cov


def lowz_sel(cov_scale=1.0):
    sn, mean, cov = get_selection_effects_skewnorm("snana_data/DES3Y_LOWZ_BHMEFF/all.npy")
    cov *= cov_scale
    return sn, mean, cov


def get_ratio(dump_npy, cut_mag=19.75):
    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)
    data = np.load(dir_name + "/" + dump_npy)
    print("Got data to compute selection function")
    mB_all = data[:, 0]
    mB_passed = mB_all[data[:, 1] > 0]

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

    threshold = 0.1
    red_chi2 = 100
    adj = 0.0001
    while np.abs(red_chi2 - 1) > threshold:
        if red_chi2 > 1:
            adj *= 1.1
        else:
            adj *= 0.9
        ratio_error_adj = ratio_error + adj
        result = curve_fit(cdf, binc, ratio, p0=np.array([23.0, 1.0, 0.0, 0.5]), sigma=ratio_error_adj)
        vals, cov, *_ = result
        chi2 = np.sum(((ratio - cdf(binc, *vals)) / ratio_error_adj) ** 2)
        red_chi2 = chi2 / (len(binc) - 3)
        print(red_chi2, adj)

    print(vals)
    print(cov)
    print(np.sqrt(np.diag(cov)))

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.errorbar(binc, ratio, yerr=ratio_error_adj, ls="none", fmt='o', ms=3)
        ax.errorbar(binc, ratio_smooth, yerr=ratio_smooth_error)

        mbs = np.linspace(binc[0], binc[-1], 1000)
        cdf_eval = cdf(mbs, *vals)

        for i in range(50):
            rands = np.random.multivariate_normal([0,0,0,0], cov=cov)
            vals2 = vals + rands
            cdf_eval2 = cdf(mbs, *vals2)
            plt.plot(mbs, cdf_eval2, c='k', alpha=0.05)

        ax.plot(mbs, cdf_eval, c='k')
        plt.show()

    return False, vals, cov


def get_selection_effects_skewnorm(dump_npy, plot=False, cut_mag=10):
    binc, ratio, ratio_error, ratio_smooth, ratio_smooth_error = get_ratio(dump_npy, cut_mag=cut_mag)

    def sknorm(b, mean, sigma, alpha, n):

        model = skewnorm.pdf(b, alpha, loc=mean, scale=sigma) * n
        return model

    threshold = 0.1
    red_chi2 = np.inf
    adj = 0.001
    while np.abs(red_chi2 - 1) > threshold:
        if red_chi2 > 1:
            adj *= 1.2
        else:
            adj *= 0.8
        ratio_error_adj = ratio_smooth_error + adj
        result = curve_fit(sknorm, binc, ratio_smooth, p0=np.array([15.0, 1.0, 0.0, 0.5]),
                           sigma=ratio_error_adj, bounds=([10, 0, -10, 0], [30, 10, 10, 5.0]))
        vals, cov, *_ = result
        chi2 = np.sum(((ratio - sknorm(binc, *vals)) / ratio_error_adj) ** 2)
        red_chi2 = chi2 / (len(binc) - 3)
        print("red chi2 ", red_chi2, adj, chi2, vals)

    print(vals)
    print(cov)
    print(np.sqrt(np.diag(cov)))

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.errorbar(binc, ratio, yerr=ratio_error_adj, ls="none", fmt='o', ms=3)
        ax.errorbar(binc, ratio_smooth, yerr=ratio_smooth_error)

        mbs = np.linspace(binc[0], binc[-1], 1000)
        cdf_eval = sknorm(mbs, *vals)

        for i in range(50):
            rands = np.random.multivariate_normal([0,0,0,0], cov=cov)
            vals2 = vals + rands
            cdf_eval2 = sknorm(mbs, *vals2)
            plt.plot(mbs, cdf_eval2, c='k', alpha=0.05)

        ax.plot(mbs, cdf_eval, c='k')
        plt.show()

    return True, vals, cov

if __name__ == "__main__":
    get_selection_effects_skewnorm("eff_data/DES3Y_LOWZ_EFF/all.npy", plot=True)
    get_selection_effects_cdf("eff_data/DES3Y_DES_EFF_AMG10/all.npy", plot=True)
    # get_selection_effects_cdf("eff_data/DES3Y_DES_EFF_CD/all.npy", plot=True)
