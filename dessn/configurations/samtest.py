import os
import logging
import socket

from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModelW, ApproximateModel
from dessn.framework.simulations.snana import SNANASimulation


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)20s()] %(message)s")
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # models = [ApproximateModelW(prior=True), ApproximateModelW(prior=True, statonly=True)]
    models = [ApproximateModelW(prior=True, statonly=True)]#, ApproximateModelW(prior=True, statonly=True, apply_efficiency=False)]
    # models = [ApproximateModelW(prior=True, statonly=True)]

    ndes = 204
    nlowz = 137
    simulations = [
            [SNANASimulation(ndes, "DES3YR_DES_SAMTEST_MAGSMEAR"), SNANASimulation(nlowz, "DES3YR_LOWZ_SAMTEST_MAGSMEAR")],
            # [SNANASimulation(ndes, "DES3YR_DES_BULK_G10_SYM"), SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_G10_SYM")],
            # [SNANASimulation(ndes, "DES3YR_DES_BULK_C11_SKEW"), SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_C11_SKEW")],
            # [SNANASimulation(ndes, "DES3YR_DES_BULK_C11_SYM"), SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_C11_SYM")]
        ]
    fitter = Fitter(dir_name)

    # data = models[0].get_data(simulations[0], 0, plot=True)  # For testing
    # print(data.keys())
    # print(data["mean_mass"])
    # print(data["masses"])
    # exit()

    fitter.set_models(*models)
    fitter.set_simulations(*simulations)
    ncosmo = 100
    fitter.set_num_cosmologies(ncosmo)
    fitter.set_max_steps(3000)
    fitter.set_num_walkers(1)
    fitter.set_num_cpu(500)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        res = fitter.load(split_models=True, split_sims=True, squeeze=False)
        # res2 = fitter.load(split_models=True, split_sims=False)

        c1, c2, c3 = ChainConsumer(), ChainConsumer(), ChainConsumer()

        for m, s, ci, chain, truth, weight, old_weight, posterior in res:
            name = s[0].simulation_name.replace("DES3YR_DES_BULK_", "").replace("_", " ").replace("SKEW", "SK16")
            name = "%s %s" % (name, m.statonly)

            if isinstance(m, ApproximateModelW):
                print("C2")
                c2.add_chain(chain, weights=weight, posterior=posterior, name=name)
            else:
                print("C1")
                c1.add_chain(chain, weights=weight, posterior=posterior, name=name)
        # for m, s, chain, truth, weight, old_weight, posterior in res2:
        #     name = "All"
        #     if isinstance(m, ApproximateModelW):
        #         print("C2")
        #         c2.add_chain(chain, weights=weight, posterior=posterior, name=name)
        #     else:
        #         print("C1")
        #         c1.add_chain(chain, weights=weight, posterior=posterior, name=name)
        #
        c2.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, shade=True)
        c2.plotter.plot_summary(filename=pfn + "2.png", parameters=["$w$"], truth=[-1.0], figsize=1.5, errorbar=True)
        c2.plotter.plot(filename=pfn + "_big.png", parameters=10, truth=truth)
        c2.plotter.plot(filename=pfn + "_big2.png", parameters=31, truth=truth)
        c2.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=7)

        # c2.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False)

        res3 = fitter.load(split_models=True, split_sims=True, split_cosmo=True, squeeze=False)
        wdict = {}
        for m, s, ci, chain, truth, weight, old_weight, posterior in res3:
            if isinstance(m, ApproximateModelW):
                name = s[0].simulation_name.replace("DES3YR_DES_BULK_", "").replace("_", " ").replace("SKEW", "SK16")
                name = "%s %s" % (name, m.statonly)
                if wdict.get(name) is None:
                    wdict[name] = []
                wdict[name].append([ci, chain])
        import numpy as np
        with open(pfn + "_comp.txt", 'w') as f:
            f.write("%10s %5s(%5s) %5s %5s\n" % ("Key", "<w>", "<werr>", "std<w>", "bias"))
            for key in sorted(wdict.keys()):
                ws = [cc[1]["$w$"] for cc in wdict[key]]
                indexes = [cc[0] for cc in wdict[key]]
                means = [np.mean(w) for w in ws]
                stds = [np.std(w) for w in ws]
                name2 = pfn + key.replace(" ", "_") + ".txt"
                with open(name2, "w") as f2:
                    for i in range(ncosmo):
                        if i in indexes:
                            f2.write("%0.5f\n" % means[indexes.index(i)])
                        else:
                            f2.write("0\n")

                # import matplotlib.pyplot as plt
                # plt.hist(means, bins=50)
                # plt.show()
                # mean_mean = np.mean(means)
                mean_mean = np.average(means, weights=1 / (np.array(stds) ** 2))
                mean_std = np.mean(stds)
                bias = (mean_mean + 1) / mean_std
                f.write("%10s %0.4f(%0.4f) %0.4f %0.4f\n" % (key, mean_mean, mean_std, np.std(means), bias))



        # print("Saving Parameter values")
        # with open(pfn + "_all_params1.txt", 'w') as f:
        #     f.write(c1.analysis.get_latex_table(transpose=True))
        # with open(pfn + "_all_params2.txt", 'w') as f:
        #     f.write(c2.analysis.get_latex_table(transpose=True))
        # c1.plotter.plot_summary(filename=pfn + "1.png", parameters=1, truth=[0.3], figsize=1.5, errorbar=True)
        # c2.plotter.plot_summary(filename=pfn + "2.png", parameters=["$w$"], truth=[-1.0], figsize=1.5, errorbar=True)

