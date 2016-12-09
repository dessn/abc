import os
from dessn.models.d_simple_stan.load import plot_quick, plot_all_weight, plot_single_cosmology_weight, \
    load_stan_from_folder
import matplotlib.pyplot as plt
import numpy as np
from chainconsumer import ChainConsumer


def debug_plots(std):
    print(std)

    res = load_stan_from_folder(std, merge=True, cut=False)
    chain, posterior, t, p, f, l, w, ow = res

    c = ChainConsumer()
    # c.add_chain(chain, weights=w, name="calib")
    c.add_chain(chain, name="calib")
    c.configure_bar(summary=True)
    c.plot(filename="output.png", parameters=9, figsize=1.3)
    c.plot(filename="output2.png", parameters=6, figsize=1.3)

    # c = ChainConsumer()
    # c.add_chain(chain, name="approx")
    # c.add_chain(chain, weights=w, name="full")
    # c.plot(filename="output.png", truth=t)

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"
    plot_quick(std, "snana", include_sep=False)
    plot_all_weight(std, dir_name + "/plot_approx_weight.png")
    #for i in range(20):
    #    plot_single_cosmology_weight(std, dir_name + "/plot_approx_single_weight_%d.png" % i, i=i)
    debug_plots(std)

