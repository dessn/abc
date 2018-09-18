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
    simulation = [SNANASimulation(-1, "DES3YR_LOWZ_COMBINED_TEXT_v8"),
                  SNANASimulation(-1, "DES3YR_DES_COMBINED_TEXT_v8")]

    fitter = Fitter(dir_name)

    # data = models[0].get_data(simulation, 0)  # For testing
    # exit()

    fitter.set_models(*models)
    fitter.set_simulations(simulation)
    fitter.set_num_cosmologies(1)
    fitter.set_max_steps(4000)
    fitter.set_num_walkers(500)
    fitter.set_num_cpu(500)

    blind = False

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

        import numpy as np
        # data = np.vstack(chain[x] for x in ['$\\Omega_m$', '$w$', '$\\alpha$', '$\\beta$', '$\\langle M_B \\rangle$', '$\\delta(0)$'])
        # data = data.T
        # np.save("samchain.npy", data.astype(np.float32))
        # np.savetxt("samchain.txt", data)
        # print(data.shape)
        # exit()

        chain_planck, params, weights, likelihood = get_planck()
        if blind:
            chain_planck[:, 0] = blind_om(chain_planck[:, 0])
            chain_planck[:, 1] = blind_w(chain_planck[:, 1])
        c.add_chain(chain_planck, parameters=params, name="Planck")

        subset = np.vstack((chain_full[r"$\Omega_m$"], chain_full["$w$"]))
        kde = gaussian_kde(chain_planck.T)
        nn = 10
        print("Trained")
        weights = kde.evaluate(subset[:, ::nn])
        print("Eval")
        c.add_chain(subset.T[::nn, :], weights=weights, name="Combined", parameters=params)
        c2.add_chain(subset.T[::nn, :], weights=weights, name="Combined", parameters=params)

        parameters = [r"$\Omega_m$", "$w$"]
        extents = {r"$\Omega_m$": (0.15, 0.65), "$w$": (-1.8, -0.5)}
        watermark = "Blinded" if blind else None

        c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, plot_hists=False,
                    sigmas=[0, 1, 2], linestyles=["-", "--", ':', '-', '--', '-'], kde=2.0,
                    legend_kwargs={"loc": "center right"}, watermark_text_kwargs={"alpha": 0.2},
                    colors=["b", "k", 'a', 'g', 'r', 'lb'], shade_alpha=[0.5, 0.0, 0.2, 0.4, 0.8, 0.1, 0.1])
        c.plotter.plot(filename=[pfn + ".png", pfn + ".pdf"], parameters=parameters, watermark=watermark,
                       figsize=1.5, extents=extents)

        c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, plot_hists=False,
                    sigmas=[0, 1, 2], linestyles=["-", "--", ':', '-', '--', '-'], kde=0.5,
                    legend_kwargs={"loc": "center right"}, watermark_text_kwargs={"alpha": 0.2},
                    colors=["p", "k", 'a', 'g', 'r', 'lb'], shade_alpha=[0.5, 0.0, 0.2, 0.4, 0.8, 0.1, 0.1])
        c.plotter.plot(filename=[pfn + "2.png", pfn + "2.pdf"], parameters=parameters, watermark=watermark,
                       figsize=1.5, extents=extents)

        with open(pfn + "_res.txt", "w") as f:
            p = c.analysis.get_latex_table(transpose=True)
            f.write(p)
        c2.plotter.plot(filename=pfn + "_big.png", parameters=17, watermark=watermark)


            # c.configure(sigma2d=False, plot_hists=True, linestyles=["-", "--", '-', ':', '-', '-'],
        #             colors=["b", "k", 'a', 'r', 'g', 'lb'], shade_alpha=[0.7, 0.0, 0.2, 0.1, 0.1, 0.1, 0.1])
        # c.plotter.plot(filename=pfn + "big.png", parameters=20)
        # c.plotter.plot_distributions(filename=pfn + "_dist.png", col_wrap=8)
        # print("Plotting distributions")
        # c = ChainConsumer()
        # c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        # c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False)
        # c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=8)
        # c2.configure(statistics="mean")
        # with open(pfn + "_nusiance_mean.txt", "w") as f:
        #     f.write(c2.analysis.get_latex_table(transpose=True))
        # c2.configure(statistics="max")
        # with open(pfn + "_nusiance_max.txt", "w") as f:
        #     f.write(c2.analysis.get_latex_table(transpose=True))
