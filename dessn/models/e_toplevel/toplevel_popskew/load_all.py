import os

from chainconsumer import ChainConsumer

from dessn.models.e_toplevel.load import plot_quick, plot_debug, plot_separate, load_stan_from_folder

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"

    c = ChainConsumer()

    datas = ["snana_dummy", "snana_diff", "snana_diff2", "simple"]
    labels = [r"SNANA $\Omega_m=0.3$", r"SNANA $\Omega_m=0.2$", r"SNANA $\Omega_m=0.4$", "Simple model with mass"]

    for data_source, label in zip(datas, labels):

        chain, posterior, t, p, f, l, w, ow = load_stan_from_folder(dir_name + "/stan_output_%s" % data_source,
                                                                cut=False, merge=True, mod_weight=False)

        c.add_chain(chain, posterior=posterior, name=label)

    c.configure(shade=True)
    c.plot(filename=dir_name + "/colour_all.png", truth=t, parameters=7)


