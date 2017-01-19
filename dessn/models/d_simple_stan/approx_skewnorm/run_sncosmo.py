import os
from scipy.interpolate import interp1d
import numpy as np

from dessn.models.d_simple_stan.load_correction_data import load_correction_supernova
from dessn.models.d_simple_stan.run import run, add_weight_to_chain, get_analysis_data


def get_approximate_mb_correction(correction_source):
    d = load_correction_supernova(correction_source=correction_source, only_passed=False)
    mask = d["passed"] == 1
    mB = d["apparents"]
    c = d["colours"]
    x1 = d["stretches"]
    alpha = 0.14
    beta = 3.1
    bins = np.linspace(20, 28, 100)
    hist_all, _ = np.histogram(mB, bins=bins)
    hist_passed, _ = np.histogram(mB[mask], bins=bins)
    binc = 0.5 * (bins[:-1] + bins[1:])
    hist_all[hist_all == 0] = 1
    ratio = 1.0 * hist_passed / hist_all
    ratio /= ratio.max()

    inter = interp1d(ratio, binc)
    mean = inter(0.5)
    width = 0.5 * (inter(0.16) - inter(0.84))
    print(mean, width)

    from astropy.cosmology import FlatLambdaCDM
    data = {
        "data_source": "sncosmo",
        "n": 500
    }
    data = get_analysis_data(**data)
    print(data["redshifts"].max())
    zs = np.linspace(0.3, 1.0, 20)
    mB_means = -19.36 + FlatLambdaCDM(70, 0.3).distmod(zs).value
    mB_mean = mB[mask].mean()
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    plt.hist(d["redshifts"][mask], 100)
    plt.show()
    plt.plot(binc, ratio)
    plt.plot(binc, 1 - norm.cdf(binc, mean, width))
    plt.plot(binc, hist_all / hist_all.max())
    plt.plot(binc, hist_passed / hist_all.max())
    for m in mB_means:
        plt.axvline(m)
    plt.axvline(mB_mean)
    plt.xlabel("mB")
    plt.ylabel("Efficiency")
    plt.show()
    exit()

    return mean, width


if __name__ == "__main__":

    file = os.path.abspath(__file__)
    stan_model = os.path.dirname(file) + "/model.stan"

    mB_mean, mB_width = get_approximate_mb_correction("sncosmo")

    data = {
        "mB_mean": mB_mean,
        "mB_width": mB_width,
        "data_source": "sncosmo",
        "n": 500
    }
    print("Running %s" % file)
    run(data, stan_model, file, weight_function=add_weight_to_chain)
