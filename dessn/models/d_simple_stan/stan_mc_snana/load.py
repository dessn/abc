import os
from dessn.models.d_simple_stan.load import plot_quick

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"
    plot_quick(std, "stan_mc", include_sep=False)
