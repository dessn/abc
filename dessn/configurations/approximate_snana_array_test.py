import os
import logging
import socket
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModel
from dessn.framework.simulations.snana import SNANASimulationSkewed0p3, SNANASimulationGauss0p2, \
    SNANASimulationGauss0p3, SNANASimulationGauss0p4

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
    simulations = [SNANASimulationSkewed0p3(), SNANASimulationGauss0p2(), SNANASimulationGauss0p3(), SNANASimulationGauss0p4()]

    fitter = Fitter(dir_name)
    fitter.set_models(model)
    fitter.set_simulations(*simulations)
    fitter.set_num_walkers(6)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        chain, truth, weight, old_weight, posterior = fitter.load()
        c = ChainConsumer()
        c.add_chain(chain, posterior=posterior, name="Stan")
        c.add_chain(chain, weights=weight, posterior=posterior, name="Corrected")
        c.plot(filename=plot_filename, truth=truth)
