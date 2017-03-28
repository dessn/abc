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
