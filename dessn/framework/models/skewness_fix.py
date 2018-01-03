# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:44:13 2017

@author: shint1
"""
import numpy as np
from scipy.stats import norm, skewnorm
from scipy.integrate import simps
from scipy.ndimage.filters import gaussian_filter
from astropy.cosmology import wCDM


def get_selection_cdf(mbs, vals):
    mean, sigma, alpha, normv = vals
    return normv * (1 - norm.cdf(mbs, mean, sigma))


def get_selection_skewnorm(mbs, vals):
    mean, sigma, alpha, normv = vals
    return normv * skewnorm.pdf(mbs, alpha, mean, sigma)


def get_approx_efficiency(dist_mod, alpha, vals, correction_skewnorm):
    n = 40000
    MBs = np.random.normal(loc=-19.365, scale=0.1, size=n)
    x1s = np.random.normal(loc=0, scale=1.0, size=n)
    cs = skewnorm.rvs(alpha, loc=0, scale=0.1, size=n)

    alphax1 = 0.14
    beta = 3.1
    mbs = MBs + dist_mod + alphax1 * x1s - beta * cs

    hist, bin_edge = np.histogram(mbs, bins=1000, normed=True)
    hist2 = gaussian_filter(hist, 15)
    bin_center = 0.5 * (bin_edge[1:] + bin_edge[:-1])

    if correction_skewnorm:
        ratio = get_selection_skewnorm(bin_center, vals)
    else:
        ratio = get_selection_cdf(bin_center, vals)

    area1 = simps(hist2, x=bin_center)
    effective = ratio * hist2
    area2 = simps(effective, x=bin_center)

    # import matplotlib.pyplot as plt
    # plt.plot(bin_center, hist2, ls="--")
    # plt.plot(bin_center, ratio, ls="-")
    # plt.plot(bin_center, effective, ls=":")

    return area2 / area1


def get_shift_scale(redshifts, correction_skewnorm, vals, plot=False):
    print("Getting shift scale", correction_skewnorm, vals, redshifts.mean())
    cosmo = wCDM(70, 0.3, 0.7)

    dist_mod = cosmo.distmod(redshifts).value

    biases = []
    if plot:
        alphas = np.logspace(0, 0.6, 5)
    else:
        alphas = np.array([1, 5])

    for alpha in alphas:
        bias_actual = np.array([get_approx_efficiency(dm, alpha, vals, correction_skewnorm) for dm in dist_mod])
        bias_computed = np.array([get_approx_efficiency(dm, 0, vals, correction_skewnorm) for dm in dist_mod])

        bias_diff = np.log(bias_actual) - np.log(bias_computed)
        total_bias = np.sum(bias_diff)
        biases.append(total_bias)

    biases = np.array(biases)
    b = biases - np.min(biases)

    fn = alphas / np.sqrt(1 + alphas ** 2)
    scale = (b[-1] - b[0]) / (fn.max() - fn.min())

    func = scale * fn

    if plot:
        import matplotlib.pyplot as plt
        for a, x in zip(alphas, b):
            print(a, x)
        func -= func.min()
        plt.figure()
        plt.plot(alphas, b)
        plt.plot(alphas, func)
        plt.show()

    return scale










