import os
import inspect
import logging
import re
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModel
from chainconsumer import ChainConsumer
from dessn.framework.simulations.snana_bulk import SNANABulkSimulation
from dessn.framework.simulations.selection_effects import lowz_sel, des_sel

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    model = ApproximateModel()
    simulation = [SNANABulkSimulation(300, sim="SHINTON_LOWZ_MATRIX_SMEAR_SYMC_SYMX1", manual_selection=lowz_sel(), num_calib=58),
                  SNANABulkSimulation(250, sim="SHINTON_DES_MATRIX_SMEAR_SYMC_SYMX1", manual_selection=des_sel(), num_calib=22)]

    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)

    expression = re.compile("approximate_bulk_(.*)_test.py")
    matches = [re.match(expression, f) for f in sorted(os.listdir(dir_name))]
    names = [m.group(1).replace("_", " ").title().replace("Gauss", "Gaussian").replace("Skew", "SK16") for m in matches if m is not None]
    filenames = [m.string[:-3] for m in matches if m is not None]

    dir_names = [os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/output" % f for f in filenames]

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
    c.configure(diagonal_tick_labels=False, linestyles=ls, colors=colors, shade_alpha=alphas, shade=True)
    params = ['$\\Omega_m$', '$\\alpha$', '$\\beta$']
    c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=8)
    c.plotter.plot(filename=[pfn + ".png", pfn + ".pdf"], parameters=params, truth=truth, figsize="column")
    with open(pfn + ".txt", 'w') as f:
        f.write(c.analysis.get_latex_table(parameters=params))
    c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], parameters=params, errorbar=True, truth=truth, extra_parameter_spacing=1.0)
