import os
import logging
import socket

from dessn.blind.blind import blind_om, blind_w
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModelW
from dessn.framework.simulations.snana_bulk import SNANABulkSimulation
from dessn.framework.simulations.selection_effects import lowz_sel, des_sel


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)
    print(dir_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    model = ApproximateModelW()
    # Turn off mass and skewness for easy test
    simulation = [SNANABulkSimulation(152, sim="PS1_LOWZ_COMBINED_FITS", manual_selection=lowz_sel(), num_calib=50),
                  SNANABulkSimulation(208, sim="DESALL_specType_SMP_real_snana_text", manual_selection=des_sel(), num_calib=21)]

    fitter = Fitter(dir_name)
    fitter.set_models(model)
    fitter.set_simulations(simulation)
    fitter.set_num_cosmologies(100)
    fitter.set_num_walkers(1)
    fitter.set_max_steps(5000)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        m, s, chain, truth, weight, old_weight, posterior = fitter.load()
        chain[r"$\Omega_m$"] = blind_om(chain[r"$\Omega_m$"])
        chain["$w$"] = blind_w(chain["$w$"])

        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="BHM")

        # Add Bonnies fits
        import numpy as np
        bonnie_dirname = os.path.dirname(os.path.abspath(__file__)) + "/chains/bz_temp/Chains-wCDMprior-DES+nearby-20171106"
        b_chain = np.loadtxt(bonnie_dirname + "/params.txt", delimiter=",")
        i = int(0.7 * b_chain.shape[0])
        b_chain = b_chain[i:, :]
        b_params = ['$\\Omega_m$', "$w$", '$\\alpha$', '$\\beta$', '$\\langle M_B \\rangle$', r"$\Delta M$"]
        c.add_chain(b_chain, parameters=b_params, name="JLA-like")

        c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, plot_hists=False, sigmas=[0, 1, 2],
                    contour_labels="confidence", kde=[False, 2.0])

        parameters = [r"$\Omega_m$", "$w$"]  # r"$\alpha$", r"$\beta$", r"$\langle M_B \rangle$"]
        print(c.analysis.get_latex_table(transpose=True))
        c.plotter.plot(filename=pfn + ".png", truth=truth, parameters=parameters, watermark="Blinded", figsize=1.5,
                       extents={r"$\Omega_m$": [0.15, 0.8], "$w$": [-1.5, -0.4]})
        # c.plotter.plot_walks(display=True)
        # print("Plotting distributions")
        # c = ChainConsumer()
        # c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        # c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False)
        # c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=8)

