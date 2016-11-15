import os
from dessn.models.d_simple_stan.load import plot_quick, plot_all_weight, plot_single_cosmology_weight, \
    load_stan_from_folder
import matplotlib.pyplot as plt
import numpy as np
from chainconsumer import ChainConsumer


def debug_plots(std):
    print(std)

    res = load_stan_from_folder(std, merge=False, num=0, cut=False)
    chain, posterior, t, p, f, l, w, ow = res[0]

    logw = np.log10(w)
    chain["ow"] = np.log10(ow)

    # tosort = chain[r"$\delta \mathcal{Z}_0$"] + chain[r"$\delta \mathcal{Z}_1$"] + chain[r"$\delta \mathcal{Z}_2$"] + chain[r"$\delta \mathcal{Z}_3$"]
    tosort = chain[r"$\delta \mathcal{Z}_1$"]
    a = np.argsort(tosort)

    logw = logw[a]
    for k in chain:
        chain[k] = chain[k][a]

    chain["ww"] = logw
    c = ChainConsumer()
    # c.add_chain(chain, weights=w, name="calib")
    c.add_chain(chain, name="calib")
    # c.plot_walks(chain="new", truth=t, filename="walk_new.png")
    res2 = load_stan_from_folder(std + "_no_calib", merge=False, cut=False)
    chain, posterior, t, p, f, l, w, ow = res2[0]
    chain["ww"] = np.log10(w)
    chain["ow"] = np.log10(ow)
    # c.add_chain(chain, weights=w, name="nocalib")
    c.add_chain(chain,  name="nocalib")
    c.plot(filename="output.png", truth=t)

    # c = ChainConsumer()
    # c.add_chain(chain, name="approx")
    # c.add_chain(chain, weights=w, name="full")
    # c.plot(filename="output.png", truth=t)

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"
    # plot_quick(std, "approx", include_sep=True)
    #plot_all_weight(std, dir_name + "/../output/plot_approx_weight.png")
    # for i in range(15):
    #     plot_single_cosmology_weight(std, dir_name + "/../output/plot_approx_single_weight_%d.png" % i, i=i)
    debug_plots(std)

