import os
from dessn.models.d_simple_stan.load import plot_quick, plot_all_no_weight

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"
    plot_quick(std, "simple", include_sep=True)
    plot_all_no_weight(std, dir_name + "/../output/plot_simple_no_weight.png")
