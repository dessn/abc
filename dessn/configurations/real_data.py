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
        print("Trained")
        weights = kde.evaluate(subset[:, ::1])
        print("Eval")
        c.add_chain(subset.T[::1, :], weights=weights, name="Combined", parameters=params)
        c2.add_chain(subset.T[::1, :], weights=weights, name="Combined", parameters=params)

        if False:
            bonnie_file = plot_dir + "/output/bonnie.txt"
            import numpy as np
            bnames = ['$\\Omega_m$', '$w$', '$\\alpha$', '$\\beta$', '$\\langle M_B \\rangle$', '$\\delta(0)$']
            bonnie_data = np.loadtxt(bonnie_file, delimiter=",")
            bonnie_data = bonnie_data[10000:, :]
            bdic = {bnames[i]: bonnie_data[:, i] for i in range(len(bnames))}
            subset_bonnie = np.vstack((bdic[r"$\Omega_m$"], bdic["$w$"]))
            c.add_chain(chain=bdic, name="JLA-Like")
            weights_bonnie = kde.evaluate(subset_bonnie[:, ::5])
            c.add_chain(subset_bonnie.T[::5, :], weights=weights_bonnie, name="Combined JLA", parameters=params)
            c2.add_chain(subset_bonnie.T[::5, :], weights=weights_bonnie, name="Combined JLA", parameters=params)

        c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, plot_hists=False,
                    sigmas=[0, 1, 2], linestyles=["-", "--", ':', '-', '--', '-'],
                    legend_kwargs={"loc": "center right"}, watermark_text_kwargs={"alpha": 0.2},
                    colors=["b", "k", 'a', 'g', 'r', 'lb'], shade_alpha=[0.5, 0.0, 0.2, 0.4, 0.8, 0.1, 0.1])
        parameters = [r"$\Omega_m$", "$w$"]  # r"$\alpha$", r"$\beta$", r"$\langle M_B \rangle$"]
        extents = {r"$\Omega_m$": (0.15, 0.65), "$w$": (-1.8, -0.5)}
        # print(c.analysis.get_latex_table(transpose=True))
        c.plotter.plot(filename=pfn + ".png", parameters=parameters, watermark="Blinded", figsize=1.5, extents=extents)
        with open(pfn + "_res.txt", "w") as f:
            p = c.analysis.get_latex_table(transpose=True)
            f.write(p)
        c2.plotter.plot(filename=pfn + "_big.png", parameters=18, watermark="Blinded")


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
