import os

from chainconsumer import ChainConsumer

from dessn.models.e_toplevel.load import plot_quick, plot_debug, plot_separate, load_stan_from_folder

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"

    c = ChainConsumer()

    datas = ["snana_dummy", "snana_diff", "snana_diff2", "simple"]
    labels = [r"SNANA $\Omega_m=0.3$", r"SNANA $\Omega_m=0.2$", r"SNANA $\Omega_m=0.4$", "Simple model with mass"]
    oms = [0.3, 0.2, 0.4, 0.3]

    for data_source, label, om in zip(datas, labels, oms):

        chain, posterior, t, p, f, l, w, ow = load_stan_from_folder(dir_name + "/stan_output_%s" % data_source,
                                                                cut=False, merge=True, mod_weight=False)
        print(list(chain.keys()))
        chain[r"$\Delta \Omega_m$"] = chain['$\\Omega_m$'] - om
        del chain['$\\Omega_m$']
        c.add_chain(chain, posterior=posterior, name=label)

    c.configure(shade=True)
    parameters = [r"$\Delta \Omega_m$", '$\\alpha$', '$\\beta$', '$\\langle M_B \\rangle$', '$\\sigma_{\\rm m_B}$',
                  '$\\sigma_{x_1}$', '$\\sigma_c$']
    t[r"$\Delta \Omega_m$"] = 0
    c.plot(filename=dir_name + "/colour_all.png", truth=t, parameters=parameters, figsize=1.5)


