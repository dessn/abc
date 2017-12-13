import os
import logging
import socket
import numpy as np
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
        from chainconsumer import ChainConsumer
        c = ChainConsumer()
        parameters = [r"$\Omega_m$", "$w$"]

        loc = "BBC_NOMINAL/NOMINAL-%04d/DB17_V1_sn_omw_%d.py_mcsamples"
        for j, name in enumerate(["BBC Stat+Syst", "BBC Stat"]):
            bbc_chain1 = np.hstack(tuple([src.read(loc % (i + 1, j), blind=False) for i in range(10)])).T
            c.add_chain(bbc_chain1[:, :2], parameters=parameters, weights=bbc_chain1[:, 2], name=name)

        res = fitter.load()

        for m, s, chain, truth, weight, old_weight, posterior in res:
            name = "BHM Stat+Syst" if not m.statonly else "BHM Stat"
            c.add_chain(chain, weights=weight, posterior=posterior, name=name)
        c1, c2 = 'lg', 'blue'
        c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, bins=0.7,
                    plot_hists=False, sigmas=[0, 1, 2], colors=[c1, c1, c2, c2],
                    linestyles=["-", "--", "-", "--"], shade_alpha=0.2, shade=True)

        print(c.analysis.get_latex_table(transpose=True))
        ns = c._names
        extents = {r"$\Omega_m$": [0.1, 0.65], "$w$": [-1.9, -0.5]}
        c.plotter.plot(filename=pfn + "Syst.png", truth=truth, parameters=parameters, figsize=1.5,
                       extents=extents, chains=[n for n in ns if "Syst" in n])
        c.plotter.plot(filename=pfn + "Stat.png", truth=truth, parameters=parameters, figsize=1.5,
                       extents=extents, chains=[n for n in ns if "Syst" not in n])
