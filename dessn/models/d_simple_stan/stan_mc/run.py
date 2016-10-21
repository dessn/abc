import os
from dessn.models.d_simple_stan.run import run

if __name__ == "__main__":

    file = os.path.abspath(__file__)
    stan_model = os.path.dirname(file) + "/model.stan"
    data = {"add_sim": 1000}
    print("Running %s" % file)
    run(data, stan_model, file)
