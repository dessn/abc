import os
import logging
import socket

from scipy.stats import gaussian_kde

from dessn.blind.blind import blind_om, blind_w
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModelW
from dessn.framework.simulations.snana import SNANASimulation
from dessn.planck.planck import get_planck

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)20s()] %(message)s")
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    models = [ApproximateModelW(), ApproximateModelW(statonly=True)]
    simulation = [SNANASimulation(-1, "DES3YR_LOWZ_COMBINED_TEXT"),
                  SNANASimulation(-1, "DES3YR_DES_COMBINED_TEXT")]

    fitter = Fitter(dir_name)

    # data = model.get_data(simulation, 0)  # For testing
    # exit()

    fitter.set_models(*models)
    fitter.set_simulations(simulation)
    fitter.set_num_cosmologies(1)
    fitter.set_max_steps(3000)
    fitter.set_num_walkers(200)
    fitter.set_num_cpu(500)

    blind = True

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer

        res = fitter.load()
        c, c2 = ChainConsumer(), ChainConsumer()

        chain_full = None

        for m, s, ci, chain, truth, weight, old_weight, posterior in res:
            if blind:
                chain[r"$\Omega_m$"] = blind_om(chain[r"$\Omega_m$"])
                chain["$w$"] = blind_w(chain["$w$"])
            name = "Stat + Syst" if not m.statonly else "Stat"
            if not m.statonly:
                chain_full = chain
            c.add_chain(chain, weights=weight, posterior=posterior, name=name)
            c2.add_chain(chain, weights=weight, posterior=posterior, name=name)

        chain_planck, params, weights, likelihood = get_planck()
        if blind:
            chain_planck[:, 0] = blind_om(chain_planck[:, 0])
            chain_planck[:, 1] = blind_w(chain_planck[:, 1])
        c.add_chain(chain_planck, parameters=params, name="Planck")

        import numpy as np
        subset = np.vstack((chain_full[r"$\Omega_m$"], chain_full["$w$"]))
        print(subset.shape, chain_planck.shape)
        kde = gaussian_kde(chain_planck.T)
        print("Trained")
        weights = kde.evaluate(subset)
        print(weights.shape)
        c.add_chain(subset.T, weights=weights, name="Combined", parameters=params)
        c2.add_chain(subset.T, weights=weights, name="Combined", parameters=params)

        c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, plot_hists=False,
                    sigmas=[0, 1, 2], linestyles=["-", "--", ':', "-"], colors=["b", "k", 'a', 'g'],
                    shade_alpha=[0.7, 0.0, 0.2, 0.8])
        parameters = [r"$\Omega_m$", "$w$"]  # r"$\alpha$", r"$\beta$", r"$\langle M_B \rangle$"]
        print(c.analysis.get_latex_table(transpose=True))
        c.plotter.plot(filename=pfn + ".png", parameters=parameters, watermark="Blinded", figsize=1.5)

        c.configure(sigma2d=False, plot_hists=True, linestyles=["-", "--", ':', '-'],
                    colors=["b", "k", 'a', 'g'], shade_alpha=[0.7, 0.0, 0.2, 0.8])

        c.plotter.plot(filename=pfn + "big.png", parameters=20)
        # print("Plotting distributions")
        # c = ChainConsumer()
        # c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        # c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False)
        # c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=8)
        c2.configure(statistics="mean")
        with open(pfn + "_nusiance_mean.txt", "w") as f:
            f.write(c2.analysis.get_latex_table(transpose=True))
        c2.configure(statistics="max")
        with open(pfn + "_nusiance_max.txt", "w") as f:
            f.write(c2.analysis.get_latex_table(transpose=True))
