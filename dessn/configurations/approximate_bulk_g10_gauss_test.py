import os
import logging
import socket
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModel
from dessn.framework.simulations.snana_bulk import SNANABulkSimulation


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(os.path.abspath(__file__)) + "/output/" + os.path.basename(__file__)[:-3]
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/"
    plot_filename = plot_dir + os.path.basename(__file__)[:-3] + ".png"
    file = os.path.abspath(__file__)
    print(dir_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    model = ApproximateModel(global_calibration=0)
    # Turn off mass and skewness for easy test
    simulation = [SNANABulkSimulation(100, sim="SHINTON_LOWZ_MATRIX_G10_SYMC_SYMX1", manual_selection=[13.70+0.5, 1.363, 3.8, 0.2]),
                  SNANABulkSimulation(250, sim="SHINTON_DES_MATRIX_G10_SYMC_SYMX1", manual_selection=[22.12, 0.544, None, 1.0])]

    fitter = Fitter(dir_name)
    fitter.set_models(model)
    fitter.set_simulations(simulation)
    fitter.set_num_cosmologies(20)
    fitter.set_num_walkers(1)
    fitter.set_max_steps(5000)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        m, s, chain, truth, weight, old_weight, posterior = fitter.load()
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        c.configure(spacing=1.0)

        parameters = [r"$\Omega_m$", r"$\alpha$", r"$\beta$", r"$\langle M_B \rangle$",
                      r"$\delta(0)$", r"$\delta(\infty)/\delta(0)$"]
        print(c.analysis.get_latex_table(transpose=True))
        # c.plotter.plot(filename=plot_filename, truth=truth, parameters=parameters)
        print("Plotting distributions")
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False)
        c.plotter.plot_distributions(filename=plot_filename.replace(".png", "_dist.png"), truth=truth, col_wrap=8)
