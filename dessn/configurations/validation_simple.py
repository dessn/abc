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

    model = ApproximateModelW(prior=True, global_calibration=1)
    # Turn off mass and skewness for easy test

    ndes = 300
    nlowz = 150
    import numpy as np

    simulations = [
        [SNANASimulation(ndes, "DES3Y_DES_VALIDATION_STATONLY"),
         SNANASimulation(nlowz, "DES3Y_LOWZ_VALIDATION_STATONLY")],
        [SNANASimulation(ndes, "DES3Y_DES_VALIDATION_STATONLY", shift=np.array([0.25, 0, 0, 0])),
         SNANASimulation(nlowz, "DES3Y_LOWZ_VALIDATION_STATONLY", shift=np.array([0.0, 0.25, 0, 0]))],
        [SNANASimulation(ndes, "DES3Y_DES_VALIDATION_STATONLY", shift=np.array([0.0, 0, 0, 0])),
         SNANASimulation(nlowz, "DES3Y_LOWZ_VALIDATION_STATONLY", shift=np.array([0.0, 0.25, 0, 0]))],
        [SNANASimulation(ndes, "DES3Y_DES_VALIDATION_STATONLY", shift=np.array([0.25, 0, 0, 0])),
         SNANASimulation(nlowz, "DES3Y_LOWZ_VALIDATION_STATONLY", shift=np.array([0.0, 0.0, 0, 0]))],
    ]
    fitter = Fitter(dir_name)

    # data = model.get_data(simulations[0], 0)  # For testing
    # exit()

    fitter.set_models(model)
    fitter.set_simulations(*simulations)
    fitter.set_num_cosmologies(200)
    fitter.set_max_steps(3000)
    fitter.set_num_walkers(2)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        import numpy as np

        res = fitter.load(split_cosmo=True, split_sims=True)

        ws = {}
        for m, s, chain, truth, weight, old_weight, posterior in res:
            key = "%s - %s" % (s[0].shift, s[1].shift)
            if key not in ws:
                ws[key] = []
            ws[key].append([chain["$w$"].mean(), np.std(chain["$w$"])])

            # for key in ws.keys():
            #     vals = np.array(ws[key])
            # print(key, vals[:, 0])
        for i, key in enumerate(sorted(ws.keys())):
            vals = np.array(ws[key])
            # print("%35s %8.4f %8.4f %8.4f %8.4f"
            print("%d %8.4f %8.4f %8.4f %8.4f   %s"
                  % (i, np.average(vals[:, 0], weights=1 / (vals[:, 1] ** 2)),
                     np.std(vals[:, 0]), np.std(vals[:, 0]) / np.sqrt(100), np.mean(vals[:, 1]),
                     key))

            # chain[r"$\Omega_m$"] = blind_om(chain[r"$\Omega_m$"])
            # chain["$w$"] = blind_w(chain["$w$"])
            #
            # c, c2 = ChainConsumer(), ChainConsumer()
            # c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
            # c2.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
            # c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, plot_hists=False, sigmas=[0, 1, 2], contour_labels="confidence")
            # c2.configure(statistics="mean")
            #
            # parameters = [r"$\Omega_m$", "$w$"]  # r"$\alpha$", r"$\beta$", r"$\langle M_B \rangle$"]
            # print(c.analysis.get_latex_table(transpose=True))
            # c.plotter.plot(filename=pfn + ".png", truth=truth, parameters=parameters, watermark="Blinded", figsize=1.5)
            # print("Plotting distributions")
            # c = ChainConsumer()
            # c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
            # c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False)
            # c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=8)
