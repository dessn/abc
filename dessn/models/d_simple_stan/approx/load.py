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

    # tosort = chain[r"$\delta \mathcal{Z}_0$"] + chain[r"$\delta \mathcal{Z}_1$"] + chain[r"$\delta \mathcal{Z}_2$"] + chain[r"$\delta \mathcal{Z}_3$"]
    tosort = chain[r"$\delta \mathcal{Z}_1$"]
    a = np.argsort(logw)

    logw = logw[a]
    for k in chain:
        chain[k] = chain[k][a]

    do_weight = True
    do_walk = True
    load_second = False

    if do_walk:
        # print("owww ", ow.min(), ow.max())
        # print("www ", w.min(), w.max())
        chain["ow"] = np.log10(ow)
        chain["ww"] = logw
        c = ChainConsumer()
        c.add_chain(chain, weights=w, name="calib")
        c.plot_walks(truth=t, filename="walk_new.png")

    c = ChainConsumer()
    if do_weight:
        c.add_chain(chain, weights=w, name="calib")
    else:
        c.add_chain(chain, name="calib")

    if load_second:
        res2 = load_stan_from_folder(std + "_calib_data_no_calib_model_and_bias", num=0, merge=False, cut=False)
        chain, posterior, _, p, f, l, w, ow = res2[0]

        if do_weight:
            c.add_chain(chain, weights=w, name="nocalib")
        else:
            c.add_chain(chain, name="nocalib")
    c.plot(filename="output.png", truth=t, figsize=0.75)

    # c = ChainConsumer()
    # c.add_chain(chain, name="approx")
    # c.add_chain(chain, weights=w, name="full")
    # c.plot(filename="output.png", truth=t)

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"
    # plot_quick(std, "approx", include_sep=True)
    # plot_all_weight(std, dir_name + "/plot_approx_weight.png")
    # for i in range(15):
    #     plot_single_cosmology_weight(std, dir_name + "/plot_approx_single_weight_%d.png" % i, i=i)
    debug_plots(std)

