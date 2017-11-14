import os
import logging
import socket

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
    simulation = [SNANABulkSimulation(500, sim="SHINTON_LOWZ_MATRIX_G10_SKEWC_SKEWX1", manual_selection=lowz_sel(), num_calib=50),
                  SNANABulkSimulation(500, sim="SHINTON_DES_MATRIX_G10_SKEWC_SKEWX1", manual_selection=des_sel(), num_calib=21)]

    # d1 = model.get_data(simulation[0], cosmology_index=0)
    # d2 = model.get_data(simulation[1], cosmology_index=0)
    # import numpy as np
    # np.savetxt("lowz_cid.txt", d1["cids"].astype(np.int))
    # print(d1["cids"].astype(np.int).shape)
    # print("")
    # np.savetxt("des_cid.txt", d2["cids"].astype(np.int))
    # print(d2["cids"].astype(np.int).shape)
    # exit()

    fitter = Fitter(dir_name)
    fitter.set_models(model)
    fitter.set_simulations(simulation)
    fitter.set_num_cosmologies(1)
    fitter.set_num_walkers(50)
    fitter.set_max_steps(5000)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        m, s, chain, truth, weight, old_weight, posterior = fitter.load()

        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        c.configure(spacing=1.0, diagonal_tick_labels=False)

        parameters = [r"$\Omega_m$", "$w$"]  # r"$\alpha$", r"$\beta$", r"$\langle M_B \rangle$"]
        print(c.analysis.get_latex_table(transpose=True))
        c.plotter.plot(filename=pfn + ".png", truth=truth, parameters=parameters, figsize=2.0)
        print("Plotting distributions")
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False)
        c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=8)

