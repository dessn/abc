import os
import logging
import socket

from dessn.blind.blind import blind_om, blind_w
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModelW
from dessn.framework.simulations.snana import SNANASimulation


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)20s()] %(message)s")
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    model = ApproximateModelW()
    # Turn off mass and skewness for easy test

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

    fitter.set_models(model)
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
        m, s, chain, truth, weight, old_weight, posterior = fitter.load()
        chain[r"$\Omega_m$"] = blind_om(chain[r"$\Omega_m$"])
        chain["$w$"] = blind_w(chain["$w$"])

        c, c2 = ChainConsumer(), ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        c2.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, plot_hists=False, sigmas=[0, 1, 2], contour_labels="confidence")
        c2.configure(statistics="mean")

        parameters = [r"$\Omega_m$", "$w$"]  # r"$\alpha$", r"$\beta$", r"$\langle M_B \rangle$"]
        print(c.analysis.get_latex_table(transpose=True))
        c.plotter.plot(filename=pfn + ".png", truth=truth, parameters=parameters, watermark="Blinded", figsize=1.5)
        print("Plotting distributions")
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False)
        c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=8)


