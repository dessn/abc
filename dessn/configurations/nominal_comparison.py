import os
import logging
import socket
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModelW
from dessn.framework.simulations.snana import SNANASimulation
import dessn.configurations.chains.samreadchains as src


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)20s()] %(message)s")
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    models = [ApproximateModelW(), ApproximateModelW(statonly=True)]
    # Turn off mass and skewness for easy test
    simulation = [SNANASimulation(-1, "DES3Y_DES_NOMINAL"),
                  SNANASimulation(-1, "DES3Y_LOWZ_NOMINAL")]

    fitter = Fitter(dir_name)

    # data = model.get_data(simulation, 0)  # For testing

    fitter.set_models(*models)
    fitter.set_simulations(simulation)
    fitter.set_num_cosmologies(10)
    fitter.set_max_steps(2000)
    fitter.set_num_walkers(30)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:

        bbc_chain1 = src.read("BBC_NOMINAL/NOMINAL_SIM_BBC5D_STATSYS", blind=False)
        print(bbc_chain1)
        exit()

        from chainconsumer import ChainConsumer
        res = fitter.load()

        c = ChainConsumer()
        for m, s, chain, truth, weight, old_weight, posterior in res:
            name = "%s %s" % (m.__class__.__name__, s[0].simulation_name.replace("_", r"\_"))
            c.add_chain(chain, weights=weight, posterior=posterior, name=name)

        c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, plot_hists=False, sigmas=[0, 1, 2], contour_labels="confidence")

        parameters = [r"$\Omega_m$", "$w$"]

        print(c.analysis.get_latex_table(transpose=True))
        c.plotter.plot(filename=pfn + ".png", truth=truth, parameters=parameters, figsize=1.5)
