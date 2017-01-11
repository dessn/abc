import os
from dessn.models.d_simple_stan.load import plot_quick, plot_debug

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"
    data_source = "snana_dummy"
    plot_debug(std, data_source)

    # plot_quick(std, "approx", include_sep=False)


