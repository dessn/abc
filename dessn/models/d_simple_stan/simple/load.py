import os
from dessn.models.d_simple_stan.load import plot_quick, plot_debug, plot_separate

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"
    data_source = "snana_dummy"
    # plot_separate(std + "_" + data_source, data_source)
    plot_debug(dir_name, data_source, weights=False)
    # plot_quick(std, "approx", include_sep=False)


