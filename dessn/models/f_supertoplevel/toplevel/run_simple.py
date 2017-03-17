import os
import numpy as np

from dessn.models.d_simple_stan.approx_skewnorm.run import get_approximate_mb_correction
from dessn.models.d_simple_stan.run import run, add_weight_to_chain


if __name__ == "__main__":

    file = os.path.abspath(__file__)
    stan_model = os.path.dirname(file) + "/model.stan"

    mB_mean, mB_width, mB_alpha = get_approximate_mb_correction("simple")
    mB_mean, mB_width, mB_alpha = 22.5, 4, -5
    print("Mean, width and alpha of selection function are ", mB_mean, mB_width, mB_alpha)

    data = {
        "mB_mean": mB_mean,
        "mB_width": mB_width,
        "mB_alpha": mB_alpha,
        "data_source": "simple",
        "n": 250
    }
    print("Running %s" % file)
    run(data, stan_model, file)
    # run(data, stan_model, file, weight_function=add_weight_to_chain)
