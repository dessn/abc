import os
from dessn.models.d_simple_stan.load import plot_all, plot_all_weight, plot_separate, \
    plot_single_cosmology

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"
    td = dir_name + "/../output/"
    plot_name = td + "plot_complete.png"
    plot_name_single = td + "plot_complete_single.png"
    plot_name_weight = td + "plot_complete_weight.png"
    plot_name_sep = td + "plot_complete_sep.png"
    walk_name = td + "plot_complete_walk.png"

    plot_all(std, plot_name)
    plot_all_weight(std, plot_name_weight)
    # plot_separate(std, plot_name_sep)
    plot_single_cosmology(std, plot_name_single, output_walk=walk_name)
