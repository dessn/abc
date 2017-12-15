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

    models = [ApproximateModelW(prior=True), ApproximateModel()]

    ndes = 204
    nlowz = 137
    simulations = [
            [SNANASimulation(ndes, "DES3Y_DES_BULK_G10_SKEW"), SNANASimulation(nlowz, "DES3Y_LOWZ_BULK_G10_SKEW")],
            [SNANASimulation(ndes, "DES3Y_DES_BULK_G10_SYM"), SNANASimulation(nlowz, "DES3Y_LOWZ_BULK_G10_SYM")],
            [SNANASimulation(ndes, "DES3Y_DES_BULK_C11_SKEW"), SNANASimulation(nlowz, "DES3Y_LOWZ_BULK_C11_SKEW")],
            [SNANASimulation(ndes, "DES3Y_DES_BULK_C11_SYM"), SNANASimulation(nlowz, "DES3Y_LOWZ_BULK_C11_SYM")]
        ]
    fitter = Fitter(dir_name)

    # data = model.get_data(simulations[0], 0)  # For testing
    # exit()

    fitter.set_models(*models)
    fitter.set_simulations(*simulations)
    fitter.set_num_cosmologies(100)
    fitter.set_max_steps(3000)
    fitter.set_num_walkers(2)
    fitter.set_num_cpu(300)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        res = fitter.load()

        c1, c2 = ChainConsumer(), ChainConsumer()

        for m, s, chain, truth, weight, old_weight, posterior in res:
            name = s[0].simulation_name.replace("DES3Y_DES_BULK_", "").replace("_", " ").replace("SKEW", "SK16")
            name = "A"
            if isinstance(m, ApproximateModelW):
                print("C2")
                c2.add_chain(chain, weights=weight, posterior=posterior, name=name)
            else:
                print("C1")
                c1.add_chain(chain, weights=weight, posterior=posterior, name=name)
            break

        # c1.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False)
        # c2.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False)
        print(c2._chains)
        import numpy as np
        print(np.any(np.isnan(c2._chains[0])))
        # c2.plotter.plot(filename=pfn + "3.png", parameters=3, figsize=1.5)
        fig = c2.plotter.plot(parameters=3, figsize=1.5)
        fig.savefig(pfn + "sigh.png", transparent=True, pad_inches=0.05, bbox_inches="tight")
        # fig.savefig(f, bbox_inches="tight", dpi=dpi, transparent=True, pad_inches=0.05)

        # c1.plotter.plot_summary(filename=pfn + "1.png", parameters=1, figsize=1.5, errorbar=True)
        # c2.plotter.plot_summary(filename=pfn + "2.png", parameters=["$w$"], figsize=1.5, errorbar=True)
        # print("Plotting distributions")
        # c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth)


