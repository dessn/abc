import os
from scipy.interpolate import interp1d
import numpy as np

from dessn.models.d_simple_stan.approx_mass.run import get_approximate_mb_correction
from dessn.models.d_simple_stan.load_correction_data import load_correction_supernova
from dessn.models.d_simple_stan.run import run, add_weight_to_chain


if __name__ == "__main__":

    file = os.path.abspath(__file__)
    stan_model = os.path.dirname(file) + "/model.stan"

    mB_mean, mB_width = get_approximate_mb_correction("sncosmo")
    print(mB_mean, mB_width)

    data = {
        "mB_mean": mB_mean,
        "mB_width": mB_width,
        "data_source": "sncosmo",
        "n": 500
    }
    print("Running %s" % file)
    run(data, stan_model, file, weight_function=add_weight_to_chain)
