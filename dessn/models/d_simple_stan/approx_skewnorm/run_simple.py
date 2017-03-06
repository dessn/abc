import os
import numpy as np

from dessn.models.d_simple_stan.approx_skewnorm.run import get_approximate_mb_correction
from dessn.models.d_simple_stan.run import run, add_weight_to_chain


if __name__ == "__main__":

    file = os.path.abspath(__file__)
    stan_model = os.path.dirname(file) + "/model.stan"

    mB_mean, mB_width, mB_alpha = get_approximate_mb_correction("simple")
    mB_mean = 20
    mB_width = 2
    mB_alpha = -5
    print("Mean, width and alpha of selection function are ", mB_mean, mB_width, mB_alpha)

    data = {
        "mB_mean": mB_mean,
        "mB_width2": mB_width**2,
        "mB_alpha2": mB_alpha**2,
        "data_source": "simple",
        "n": 500,
        "add_zs": 501
    }
    print("Running %s" % file)
    run(data, stan_model, file, weight_function=add_weight_to_chain)
