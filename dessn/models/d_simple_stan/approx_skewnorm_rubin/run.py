import os
from scipy.interpolate import interp1d
import numpy as np
from scipy.stats import skewnorm
from dessn.models.d_simple_stan.load_correction_data import load_correction_supernova
from dessn.models.d_simple_stan.run import run, add_weight_to_chain


def get_approximate_mb_correction(correction_source):
    d = load_correction_supernova(correction_source=correction_source, only_passed=False)
    mask = d["passed"] == 1
    mB = d["apparents"]
    data = mB[mask][::10]
    print("Fitting data profile")
    alpha, mean, std = skewnorm.fit(data)

    return mean, std, alpha


if __name__ == "__main__":

    file = os.path.abspath(__file__)
    stan_model = os.path.dirname(file) + "/model.stan"

    mB_mean, mB_width, mB_alpha = get_approximate_mb_correction("snana")
    print(mB_mean, mB_width, mB_alpha)

    data = {
        "mB_mean": mB_mean,
        "mB_width2": mB_width**2,
        "mB_alpha2": mB_alpha**2,
        "data_source": "snana_dummy",
        "n": 500
    }
    print("Running %s" % file)
    run(data, stan_model, file, weight_function=add_weight_to_chain)
