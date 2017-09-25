import os
import logging
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModelOl
from dessn.framework.simulations.simple import SimpleSimulation
from chainconsumer import ChainConsumer


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/"
    d = "comparison_with_x1_c_populations"
    pfn = plot_dir + d + "/comparison.png"
    file = os.path.abspath(__file__)

    model = ApproximateModelOl(global_calibration=1)
    simulation = [SimpleSimulation(700, alpha_c=0, mass=True, dscale=0.08),
                  SimpleSimulation(300, alpha_c=0, mass=True, dscale=0.08, lowz=True)]

    dir_names = [plot_dir + d + "/with_pop", plot_dir + d + "/without_pop"]
    names = ["With Pop", "Without Pop"]

    c = ChainConsumer()

    for dir_name, name in zip(dir_names, names):
        print(dir_name)
        fitter = Fitter(dir_name)
        fitter.set_models(model)
        fitter.set_simulations(simulation)
        m, s, chain, truth, weight, old_weight, posterior = fitter.load()
        c.add_chain(chain, weights=weight, posterior=posterior, name=name)

    c.configure(diagonal_tick_labels=False, plot_hists=False, colors=["b", "k"], linestyles=["-", "--"])
    parameters = [r"$\Omega_m$", r"$\Omega_\Lambda$"]

    c.plotter.plot(filename=[pfn, pfn.replace(".png", ".pdf")], parameters=parameters,
                   figsize="column")

