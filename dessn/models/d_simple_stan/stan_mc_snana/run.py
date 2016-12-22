import os
from dessn.models.d_simple_stan.run import run
from dessn.models.d_simple_stan.snana_dummy.run import add_weight_to_chain

if __name__ == "__main__":

    file = os.path.abspath(__file__)
    stan_model = os.path.dirname(file) + "/model.stan"
    data = {
        "add_sim": 1000,
        "snana_dummy": True,
        "sim": False
    }

    print("Running %s" % file)
    run(data, stan_model, file, weight_function=add_weight_to_chain)
