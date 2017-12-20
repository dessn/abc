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

    models = [ApproximateModelW(), ApproximateModelW(statonly=True)]
    simulation = [SNANASimulation(-1, "DES3YR_LOWZ_COMBINED_FITS"),
                  SNANASimulation(-1, "DES3YR_DES_COMBINED_FITS")]

    fitter = Fitter(dir_name)

    # data = model.get_data(simulation, 0)  # For testing
    # exit()

    fitter.set_models(*models)
    fitter.set_simulations(simulation)
    fitter.set_num_cosmologies(1)
    fitter.set_max_steps(3000)
    fitter.set_num_walkers(200)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer

        res = fitter.load()
        c, c2 = ChainConsumer(), ChainConsumer()

        for m, s, chain, truth, weight, old_weight, posterior in res:
            chain[r"$\Omega_m$"] = blind_om(chain[r"$\Omega_m$"])
            chain["$w$"] = blind_w(chain["$w$"])
            name = "Stat + Syst" if not m.statonly else "Stat"
            c.add_chain(chain, weights=weight, posterior=posterior, name=name)
            c2.add_chain(chain, weights=weight, posterior=posterior, name=name)

        c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, plot_hists=False,
                    sigmas=[0, 1, 2], linestyles=["-", "--"], colors=["b", "k"], shade_alpha=[1.0, 0.0])
        c2.configure(statistics="mean")
        parameters = [r"$\Omega_m$", "$w$"]  # r"$\alpha$", r"$\beta$", r"$\langle M_B \rangle$"]
        print(c.analysis.get_latex_table(transpose=True))
        c.plotter.plot(filename=pfn + ".png", parameters=parameters, watermark="Blinded", figsize=1.5)
        # print("Plotting distributions")
        # c = ChainConsumer()
        # c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        # c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False)
        # c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=8)
        with open(pfn + "_nusiance.txt", "w") as f:
            f.write(c2.analysis.get_latex_table(transpose=True, parameters=[r"$\Omega_m$", "$w$",
                                                                            r"$\alpha$", r"$\beta$",
                                                                            r"$\sigma_{\rm m_B}^{0}$",
                                                                            r"$\sigma_{\rm m_B}^{1}$",
                                                                            r"$\delta(0)$",
                                                                            r"$\delta(\infty)/\delta(0)$"]))

