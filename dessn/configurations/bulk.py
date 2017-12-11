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
    fitter.set_num_cosmologies(25)
    fitter.set_max_steps(2000)
    fitter.set_num_walkers(5)
    fitter.set_num_cpu(300)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        res = fitter.load()

        c = ChainConsumer()
        for m, s, chain, truth, weight, old_weight, posterior in res:
            name = "%s %s" % (m.__class__.__name__, s[0].simulation_name.replace("_", r"\_"))
            c.add_chain(chain, weights=weight, posterior=posterior, name=name)

        c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, plot_hists=False, sigmas=[0, 1, 2], contour_labels="confidence")
        parameters = [r"$\Omega_m$", "$w$"]
        print(c.analysis.get_latex_table(transpose=True))
        c.plotter.plot(filename=pfn + ".png", truth=truth, parameters=parameters, watermark="Blinded", figsize=1.5)
        print("Plotting distributions")
        c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth)


