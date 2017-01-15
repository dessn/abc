import os
from scipy.interpolate import interp1d
import numpy as np

from dessn.models.d_simple_stan.load_correction_data import load_correction_supernova
from dessn.models.d_simple_stan.run import run, add_weight_to_chain


def get_approximate_mb_correction(correction_source):
    d = load_correction_supernova(correction_source=correction_source, only_passed=False)
    mask = d["passed"] == 1
    mB = d["apparents"]
    c = d["colours"]
    x1 = d["stretches"]
    alpha = 0.15
    beta = 3.0
    bins = np.linspace(19.5, mB.max(), 50)
    hist_all, _ = np.histogram(mB, bins=bins)
    hist_passed, _ = np.histogram(mB[mask], bins=bins)
    binc = 0.5 * (bins[:-1] + bins[1:])
    ratio = 1.0 * hist_passed / hist_all
    ratio /= ratio.max()

    # import matplotlib.pyplot as plt
    # plt.plot(ratio)
    # plt.show()
    # exit()

    inter = interp1d(ratio, binc)
    mean = inter(0.5)
    width = 0.5 * (inter(0.16) - inter(0.84))
    width += 0.25 * (alpha * np.std(x1) + beta * np.std(c))
    return mean, width + 0.02


if __name__ == "__main__":

    file = os.path.abspath(__file__)
    stan_model = os.path.dirname(file) + "/model.stan"

    mB_mean, mB_width = get_approximate_mb_correction("snana")
    print(mB_mean, mB_width)

    data = {
        "mB_mean": mB_mean,
        "mB_width": mB_width,
        "data_source": "snana_dummy",
        "n": 500
    }
    print("Running %s" % file)
    run(data, stan_model, file, weight_function=add_weight_to_chain)
