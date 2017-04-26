import logging
import os

from dessn.framework.models.approx_model import ApproximateModel
from dessn.framework.models.full_model import FullModel
from dessn.framework.simulations.simple import SimpleSimulation
from dessn.framework.fitter import Fitter
from chainconsumer import ChainConsumer


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/"
    plot_filename = plot_dir + os.path.basename(__file__)[:-3] + ".png"
    dir_name1 = os.path.dirname(os.path.abspath(__file__)) + "/output/full_simple_test"
    dir_name2 = os.path.dirname(os.path.abspath(__file__)) + "/output/full_snana_test"

    model1 = FullModel(500)
    model2 = ApproximateModel(500)
    simulation = SimpleSimulation()

    fitter1 = Fitter(dir_name1)
    fitter1.set_models(model1)
    fitter1.set_simulations(simulation)

    fitter2 = Fitter(dir_name2)
    fitter2.set_models(model2)
    fitter2.set_simulations(simulation)

    c = ChainConsumer()

    m, s, chain, truth, weight, old_weight, posterior = fitter1.load()
    c.add_chain(chain, posterior=posterior, name="Simple, Stan")
    c.add_chain(chain, weights=weight, posterior=posterior, name="Simple, Full")

    m, s, chain, truth, weight, old_weight, posterior = fitter2.load()
    c.add_chain(chain, posterior=posterior, name="SNANA, Stan")
    c.add_chain(chain, weights=weight, posterior=posterior, name="SNANA, Full")

    colours = [c._all_colours[0], c._all_colours[0], c._all_colours[1], c._all_colours[1]]
    c.configure(shade=True, linestyles=["-", "--", "-", "--"], colors=colours, shade_alpha=[0.3, 0, 0.3, 0])
    parameters = ['$\\Omega_m$', '$\\alpha$', '$\\beta$', '$\\langle M_B \\rangle$',
                  '$\\sigma_{\\rm m_B}$', '$\\sigma_{x_1}$', '$\\sigma_c$']

    c.plot(filename=plot_filename, truth=truth, parameters=parameters)
