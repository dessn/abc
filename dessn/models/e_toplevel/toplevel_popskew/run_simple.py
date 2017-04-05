import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import skewnorm

from dessn.models.e_toplevel.load_correction_data import load_correction_supernova
from dessn.models.e_toplevel.run import run, add_weight_to_chain


def get_approximate_mb_correction(correction_source):
    if correction_source == "simple":
        all_data = load_correction_supernova(correction_source=correction_source, only_passed=False)
        mB = np.array(all_data["apparents"])
        mask = all_data["passed"]
        data = mB[mask]
    else:
        all_data = load_correction_supernova(correction_source=correction_source, only_passed=False)
        passed_data = load_correction_supernova(correction_source=correction_source, only_passed=True)
        data = passed_data["apparents"]
        mB = all_data["apparents"]
    print("Fitting data profile")

    # Getting the efficiency pdf
    hist_all, bins = np.histogram(mB, bins=100)
    hist_passed, _ = np.histogram(data, bins=bins)
    binc = 0.5 * (bins[:-1] + bins[1:])
    hist_all[hist_all == 0] = 1
    ratio = hist_passed / hist_all

    # Inverse transformation sampling to sample from this random pdf
    cdf = ratio.cumsum()
    cdf = cdf / cdf.max()
    cdf[0] = 0
    cdf[-1] = 1
    n = 100000
    u = np.random.random(size=n)
    y = interp1d(cdf, binc)(u)

    alpha, mean, std = skewnorm.fit(y)

    # import matplotlib.pyplot as plt
    # print(mB.shape)
    # plt.plot(binc, ratio * skewnorm.pdf(mean, alpha, mean, std))
    # plt.plot(binc, skewnorm.pdf(binc, alpha, mean, std))
    # plt.hist(y, 100, histtype='step', normed=True)
    # plt.show()
    # exit()

    return mean, std, alpha


if __name__ == "__main__":

    file = os.path.abspath(__file__)
    stan_model = os.path.dirname(file) + "/model.stan"

    mB_mean, mB_width, mB_alpha = get_approximate_mb_correction("simple")
    # mB_mean, mB_width, mB_alpha = 22.5, 4, -5
    #
    # import matplotlib.pyplot as plt
    # from astropy.cosmology import FlatLambdaCDM
    # from scipy.stats import norm, skewnorm
    # from scipy.integrate import simps
    # zs = np.linspace(0.05, 1.2, 200)
    # mus = FlatLambdaCDM(70, 0.3).distmod(zs).value
    # mean = -19.365 + mus
    # cor_mb_width2 = 0.1**2 + (0.14 * 1.0)**2 + (3.1 * 0.1)**2
    # mB_width2 = mB_width**2
    # mB_alpha2 = mB_alpha**2
    # cor_sigma2 = ((cor_mb_width2 + mB_width2) / mB_width2)**2 * ((mB_width2 / mB_alpha2) + ((mB_width2 * cor_mb_width2) / (cor_mb_width2 + mB_width2)))
    # weights = norm.logpdf(mean, mB_mean, np.sqrt(mB_width2 + cor_mb_width2)) + norm.logcdf(mB_mean, mean, np.sqrt(cor_sigma2))
    #
    # mags = np.linspace(15, 25, 1000)
    # p2 = skewnorm.pdf(mags, mB_alpha, mB_mean, mB_width)
    # weights2 = []
    # for m in mean:
    #     p1 = norm.pdf(mags, m, np.sqrt(cor_mb_width2))
    #     p3 = p1 * p2
    #     weights2.append(simps(p3, mags))
    #
    # plt.plot(zs, 2*np.exp(weights))
    # plt.plot(zs, weights2)
    # plt.xlabel("z")
    # plt.ylabel("weight")
    # plt.show()
    # plt.clf()
    # plt.plot(mags, skewnorm.pdf(mags, mB_alpha, mB_mean, mB_width), 'k--')
    # for m,z in zip(mean[::20], zs[::20]):
    #     plt.plot(mags, norm.pdf(mags, m, np.sqrt(cor_mb_width2)), label="z=%0.2f"%z)
    # plt.xlabel("m_B")
    # plt.ylabel("pdf")
    # plt.legend(loc=2)
    # plt.show()
    # exit()

    print("Mean, width and alpha of selection function are ", mB_mean, mB_width, mB_alpha)

    data = {
        "mB_mean": mB_mean,
        "mB_width2": mB_width**2,
        "mB_alpha2": mB_alpha**2,
        "data_source": "simple",
        "n": 500
    }
    print("Running %s" % file)
    run(data, stan_model, file)
    # run(data, stan_model, file, weight_function=add_weight_to_chain)
