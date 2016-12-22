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
    # print(w.mean())
    # import matplotlib.pyplot as plt
    # plt.hist(np.log(w), 100)
    # plt.show()
    # exit()
    logw = np.log(w)
    m = np.mean(logw)
    s = np.std(logw)
    print(m, s)
    logw -= (m + 2.5 * s)
    good = logw < 0
    logw *= good
    w = np.exp(logw)

    c = ChainConsumer()
    c.add_chain(chain, weights=w, name="corrected")
    c.configure(summary=True, sigmas=[0,1,2])
    c.plot(figsize=2.0, filename="output.png", parameters=9)

    c = ChainConsumer()
    c.add_chain(chain, name="uncorrected")
    c.add_chain(chain, weights=w, name="corrected")
    # c.add_chain(chain, name="calib")
    c.plot(filename="output_comparison.png", parameters=9, figsize=1.3)
    c.plot_walks(chains=1, filename="walks.png")
    # c.add_chain(chain, name="approx")
    # c.add_chain(chain, weights=w, name="full")
    # c.plot(filename="output.png", truth=t)

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"
    # plot_quick(std, "snana", include_sep=False)
    # plot_all_weight(std, dir_name + "/plot_approx_weight.png")
    #for i in range(20):
    #    plot_single_cosmology_weight(std, dir_name + "/plot_approx_single_weight_%d.png" % i, i=i)
    debug_plots(std)

