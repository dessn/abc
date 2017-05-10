import os
import logging
import socket
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModel
from dessn.framework.simulations.snana import SNANASimulationGauss0p3

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

    model = ApproximateModel(500)
    # Turn off mass and skewness for easy test
    simulation = SNANASimulationGauss0p3()

    fitter = Fitter(dir_name)
    fitter.set_models(model)
    fitter.set_simulations(simulation)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        m, s, chain, truth, weight, old_weight, posterior = fitter.load()
        import numpy as np
        print(old_weight.mean(), np.std(old_weight))
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior)
        # c.plot_walks(filename=plot_filename.replace(".png", "_walks.png"))
        parameters = ['$\\Omega_m$', '$\\alpha$', '$\\beta$', '$\\langle M_B \\rangle$',
                      '$\\sigma_{\\rm m_B}$', '$\\sigma_{x_1}$', '$\\sigma_c$',
                      '$\\delta(0)$', '$\\delta(\\infty)/\\delta(0)$']
        c.plot(filename=plot_filename, truth=truth, parameters=parameters)
        # c.plot(filename=plot_filename, truth=truth)
