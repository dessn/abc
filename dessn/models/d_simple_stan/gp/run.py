import os

from sklearn.gaussian_process import GaussianProcessRegressor

from dessn.models.d_simple_stan.run import run, get_gp_data
from dessn.models.d_simple_stan.snana_dummy.run import add_weight_to_chain


def get_gp(n_sne, add_gp, seed=0, alpha=0.001, correction_source="snana"):
    flat, vals, _ = get_gp_data(n_sne, add_gp, seed=seed, correction_source=correction_source)
    gp = GaussianProcessRegressor(alpha=alpha)
    gp.fit(flat, vals)
    return gp, flat, vals


def get_gp_dict(n_sne, add_gp, correction_source="snana"):
    gp, flat, _ = get_gp(n_sne, add_gp, correction_source=correction_source)
    result = {
        "n_gp": add_gp,
        "gp_points": flat,
        "gp_alpha": gp.alpha_
    }
    return result

if __name__ == "__main__":

    file = os.path.abspath(__file__)
    stan_model = os.path.dirname(file) + "/model.stan"

    data = {
        "data_source": "snana_dummy",
        "n": 500
    }
    n_gp = 500
    gp_dict = get_gp_dict(data["n"], n_gp, data["correction_source"])

    data = {**data, **gp_dict}

    print("Running %s" % file)
    run(data, stan_model, file, weight_function=add_weight_to_chain)
