import os

from chainconsumer import ChainConsumer

from dessn.models.e_toplevel.load import plot_quick, plot_debug, plot_separate, load_stan_from_folder

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"
    data_source = "snana_diff"
    # plot_separate(std + "_" + data_source, data_source, weights=False)
    # plot_debug(dir_name, data_source, sort=False)
    # plot_quick(std, "approx", include_sep=False)

    chain, posterior, t, p, f, l, w, ow = load_stan_from_folder(dir_name + "/stan_output_%s" % data_source,
                                                                cut=False, merge=True, mod_weight=False)
    t[r'$\Omega_m$'] = 0.2
    c = ChainConsumer()
    # c.add_chain(chain, weights=w, posterior=posterior, name="Corrected")
    # chain["weight"] = ow
    c.add_chain(chain, posterior=posterior, name="Corrected")
    # c.configure(color_params="weight", colors=["#a81515", "#111111"])
    c.plot(filename=dir_name + "/colour_diff.png", truth=t)


