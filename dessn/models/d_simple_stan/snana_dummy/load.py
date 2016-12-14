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
    print(w.mean(), np.std(w))
    # import matplotlib.pyplot as plt
    # plt.hist(np.log(w), 100)
    # plt.show()
    # exit()
    c = ChainConsumer()
    truth = [0.3, 0.14, 3.1, -19.365, 0, 0, 0.1, 1.0, 0.1, 0, 0, 0, 0, 0, 0]
    c.add_chain(chain, name="uncorrected", posterior=posterior)
    c.add_chain(chain, weights=w, name="corrected", posterior=posterior)
    c.plot(filename="output.png", parameters=9, truth=truth, figsize=1.3)
    # c = ChainConsumer()
    # c.add_chain(chain, weights=w, name="corrected")
    c.plot_walks(chain="corrected", filename="walks.png", truth=truth)

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"
    # plot_quick(std, "snana", include_sep=False)
    # plot_all_weight(std, dir_name + "/plot_approx_weight.png")
    #for i in range(20):
    #    plot_single_cosmology_weight(std, dir_name + "/plot_approx_single_weight_%d.png" % i, i=i)
    debug_plots(std)

