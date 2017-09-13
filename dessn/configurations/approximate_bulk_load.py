import os
import inspect
import logging
import re
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModel
from chainconsumer import ChainConsumer
from dessn.framework.simulations.snana_bulk import SNANABulkSimulation

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/"
    plot_filename = plot_dir + os.path.basename(__file__)[:-3] + ".png"
    file = os.path.abspath(__file__)

    model = ApproximateModel()
    simulation = [SNANABulkSimulation(100, sim="SHINTON_LOWZ_MATRIX_SMEAR_SYMC_SYMX1", manual_selection=[13.70+0.5, 1.363, 3.8, 0.2], num_calib=58),
                  SNANABulkSimulation(250, sim="SHINTON_DES_MATRIX_SMEAR_SYMC_SYMX1", manual_selection=[22.12, 0.544, None, 1.0], num_calib=22)]

    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)

    expression = re.compile("approximate_bulk_(.*)_test.py")
    matches = [re.match(expression, f) for f in sorted(os.listdir(dir_name))]
    names = [m.group(1).replace("_", " ") for m in matches if m is not None]
    filenames = [m.string[:-3] for m in matches if m is not None]

    dir_names = [os.path.dirname(os.path.abspath(__file__)) + "/output/" + f for f in filenames]

    c = ChainConsumer()

    for dir_name, filename, name in zip(dir_names, filenames, names):
        print(dir_name)
        fitter = Fitter(dir_name)
        fitter.set_models(model)
        fitter.set_simulations(simulation)
        fitter.set_num_cosmologies(200)
        fitter.set_num_walkers(1)
        fitter.set_max_steps(5000)
        try:
            m, s, chain, truth, weight, old_weight, posterior = fitter.load()
            c.add_chain(chain, weights=weight, posterior=posterior, name=name)
        except UnboundLocalError:
            print("%s not get generated" % name)

    ls = ["-", ":"] * 3
    colors = ['b', 'b', 'r', 'r', 'g', 'g']
    alphas = 0.1
    c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False, linestyles=ls,
                colors=colors, shade_alpha=alphas, shade=True)
    c.plotter.plot_distributions(filename=plot_filename.replace(".png", "_dist.png"), truth=truth, col_wrap=8)
    params = ['$\\Omega_m$', '$\\alpha$', '$\\beta$', '$\\langle M_B \\rangle$']
    c.plotter.plot(filename=plot_filename, parameters=params, truth=truth)
    with open(plot_filename.replace(".png", ".txt"), 'w') as f:
        f.write(c.analysis.get_latex_table(parameters=params))

