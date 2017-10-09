import os
import logging
import socket
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModelW, ApproximateModel, ApproximateModelOl
from dessn.framework.simulations.snana_bulk import SNANACombinedBulk
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

    models = ApproximateModelW(), ApproximateModelW(systematics_scale=0.001), \
             ApproximateModel(), ApproximateModel(systematics_scale=0.001), \
             ApproximateModelOl(), ApproximateModelOl(systematics_scale=0.001)

    # Turn off mass and skewness for easy test
    simulation = [SNANACombinedBulk(300, ["SHINTON_LOWZ_MATRIX_G10_SKEWC_SKEWX1", "SHINTON_LOWZ_MATRIX_C11_SKEWC_SKEWX1"],
                                    "CombinedLowZ", manual_selection=lowz_sel(), num_calib=58),
                  SNANACombinedBulk(250, ["SHINTON_DES_MATRIX_G10_SKEWC_SKEWX1", "SHINTON_DES_MATRIX_C11_SKEWC_SKEWX1"],
                                    "CombinedDES", manual_selection=des_sel(), num_calib=22)]

    fitter = Fitter(dir_name)
    fitter.set_models(*models)
    fitter.set_simulations(simulation)
    fitter.set_num_cosmologies(100)
    fitter.set_num_walkers(1)
    fitter.set_max_steps(3000)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        m, s, chain, truth, weight, old_weight, posterior = fitter.load()
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        c.configure(spacing=1.0)

        parameters = [r"$\Omega_m$", r"$\alpha$", r"$\beta$", r"$\langle M_B \rangle$"]
        print(c.analysis.get_latex_table(transpose=True))
        c.plotter.plot(filename=pfn + ".png", truth=truth, parameters=parameters)
        print("Plotting distributions")
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False)
        c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=8)

