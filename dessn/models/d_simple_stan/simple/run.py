import os
from dessn.models.d_simple_stan.run import run, add_weight_to_chain

if __name__ == "__main__":

    file = os.path.abspath(__file__)
    stan_model = os.path.dirname(file) + "/model.stan"
    data = {
        "data_source": "snana_dummy",
        "correction_source": "snana",
        "n": 500
    }
    print("Running %s" % file)
    run(data, stan_model, file, weight_function=add_weight_to_chain)
