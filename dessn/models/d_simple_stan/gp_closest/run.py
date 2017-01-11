import os

from dessn.models.d_simple_stan.run import run, get_gp_data, get_correction_data_from_data_source
from dessn.models.d_simple_stan.run import add_weight_to_chain


def get_gp_dict(n_sne, add_gp, correction_source="snana"):
    flat, vals, _ = get_gp_data(n_sne, add_gp, correction_source=correction_source)

    result = {
        "n_gp": add_gp,
        "gp_points": flat,
        "gp_alpha": vals
    }
    return result

if __name__ == "__main__":

    file = os.path.abspath(__file__)
    stan_model = os.path.dirname(file) + "/model.stan"

    data = {
        "data_source": "snana_dummy",
        "n": 500
    }
    n_gp = 2000
    correction_source = get_correction_data_from_data_source(data["data_source"])
    gp_dict = get_gp_dict(data["n"], n_gp, correction_source)

    data = {**data, **gp_dict}

    print("Running %s" % file)
    run(data, stan_model, file, weight_function=add_weight_to_chain)
