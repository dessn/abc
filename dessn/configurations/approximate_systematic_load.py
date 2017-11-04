import os
import logging
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModel
from dessn.framework.simulations.snana_sys import SNANASysSimulation
from dessn.framework.simulations.selection_effects import lowz_sel, des_sel

from chainconsumer import ChainConsumer

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    model = ApproximateModel(global_calibration=0)
    simulation = [SNANASysSimulation(300, sys_index=0, sim="lowz", manual_selection=lowz_sel()),
                  SNANASysSimulation(500, sys_index=0, sim="des", manual_selection=des_sel())]

    filenames = ["approximate_systematic_%d_test" % i for i in range(7)]
    names = ["Stat only", "ZP offset .02 mag (Gauss)", r"Filter $\Lambda$ shift 20$\textup{\AA}$ gaus",
             "10\\% Gauss error in biasCor flux errors", "idem, but with incorrect reported fluxErr",
             "MWEBV scale from 20\\% Gauss error", "MW RV shift from 0.2 Gauss error"]
    names = ["Stat only", "ZP offset", r"Filter $\lambda$ shift",
             "Bias corrected flux error", "idem + uncertainty error",
             "MW $E(B-V)$ scale error", "MW $R_V$ shift"]
    dir_names = [os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/output/" % f for f in filenames]

    c = ChainConsumer()

    for dir_name, filename, name in zip(dir_names, filenames, names):
        print(dir_name)
        fitter = Fitter(dir_name)
        fitter.set_models(model)
        fitter.set_simulations(simulation)
        m, s, chain, truth, weight, old_weight, posterior = fitter.load()
        c.add_chain(chain, weights=weight, posterior=posterior, name=name)

    ls = ["-"] + ["-"] * (len(dir_names) - 1)
    colors = ['k', 'b', 'r', 'g', 'purple', 'o', 'lb']
    alphas = [0.3] + [0.0] * (len(dir_names) - 1)
    c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False, linestyles=ls,
                colors=colors, shade_alpha=alphas, shade=True, bar_shade=True)
    print("Plotting distributions")
    c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=8)
    params = ['$\\Omega_m$', '$\\alpha$', '$\\beta$', '$\\langle M_B \\rangle$']
    print("Plotting plot")
    c.plotter.plot(filename=[pfn + ".png", pfn + ".pdf"], parameters=params)
    print("Plotting summary")
    c.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], parameters=['$\\Omega_m$'],
                           extents={'$\\Omega_m$': [0.1, 0.45]}, truth="Stat only", extra_parameter_spacing=1.5)

