import os
import logging
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModel
from dessn.framework.simulations.snana_sys import SNANASysSimulation
from chainconsumer import ChainConsumer

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/"
    plot_filename = plot_dir + os.path.basename(__file__)[:-3] + ".png"
    file = os.path.abspath(__file__)

    model = ApproximateModel(global_calibration=0)
    simulation = [SNANASysSimulation(100, sys_index=0, sim="lowz", manual_selection=[13.70+0.5, 1.363, 3.8, 0.2]),
                  SNANASysSimulation(250, sys_index=0, sim="des", manual_selection=[22.4, 0.7, None, 1.0])]

    filenames = ["approximate_systematic_%d_test" % i for i in range(7)]
    names = ["Stat only", "ZP offset .02 mag gaus", "filter lam shift 20A gaus",
             "10\\% gaus error in biasCor flux errors", "idem, but with incorrect reported fluxErr",
             "MWEBV scale from 20\\% gaus error", "MW RV shift from 0.2 gaus error"]
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

        m, s, chain, truth, weight, old_weight, posterior = fitter.load()
        c.add_chain(chain, weights=weight, posterior=posterior, name=name)

    ls = ["-"] + ["-"] * (len(dir_names) - 1)
    colors = ['k', 'b', 'r', 'g', 'purple', 'y', 'lb']
    alphas = [0.3] + [0.0] * (len(dir_names) - 1)
    c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False, linestyles=ls,
                colors=colors, shade_alpha=alphas, shade=True)
    c.plotter.plot_distributions(filename=plot_filename.replace(".png", "_dist.png"), truth=truth, col_wrap=8)
    params = ['$\\Omega_m$', '$\\alpha$', '$\\beta$', '$\\langle M_B \\rangle$']
    c.plotter.plot(filename=plot_filename, parameters=params)

