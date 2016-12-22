import os
import numpy as np
from chainconsumer import ChainConsumer
from dessn.models.d_simple_stan.load import plot_quick, load_stan_from_folder


def debug_plots(std):
    print(std)

    res = load_stan_from_folder(std, merge=True, cut=False)
    chain, posterior, t, p, f, l, w, ow = res
    print(w.mean(), np.std(w), np.mean(np.log(w)), np.std(np.log(w)))
    # import matplotlib.pyplot as plt
    # plt.hist(np.log(w), 100)
    # plt.show()
    # exit()

    logw = np.log(w)
    m = np.mean(logw)
    s = np.std(logw)
    print(m, s)
    logw -= (m + 3 * s)
    good = logw < 0
    logw *= good
    w = np.exp(logw)

    sorti = np.argsort(w)
    for key in chain.keys():
        chain[key] = chain[key][sorti]
    w = w[sorti]
    ow = ow[sorti]
    posterior = posterior[sorti]

    c = ChainConsumer()
    truth = [0.3, 0.14, 3.1, -19.365, 0, 0, 0.1, 1.0, 0.1, 0, 0, 0, 0, 0, 0]
    chain["posterior"] = posterior
    # chain["log_weight"] =np.log(w)
    c.add_chain(chain, name="uncorrected", posterior=posterior)
    # c.add_chain(chain, weights=w, name="corrected", posterior=posterior)
    c.configure(color_params="posterior")
    c.plot(filename="output.png", parameters=9, truth=truth, figsize=1.3)
    # c = ChainConsumer()
    # c.add_chain(chain, weights=w, name="corrected")
    c.plot_walks(chains="uncorrected", filename="walks.png", truth=truth)

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"
    # plot_quick(std, "stan_mc_snana", include_sep=False)
    # plot_all_weight(std, dir_name + "/plot_approx_weight.png")
    #for i in range(20):
    #    plot_single_cosmology_weight(std, dir_name + "/plot_approx_single_weight_%d.png" % i, i=i)
    debug_plots(std)