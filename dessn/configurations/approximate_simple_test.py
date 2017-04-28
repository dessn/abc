import os
import logging
import socket
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModel
from dessn.framework.simulations.simple import SimpleSimulation

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

    num_nodes = 1
    model = ApproximateModel(300, num_nodes=num_nodes)
    simulation = SimpleSimulation(alpha_c=0, mass=False, dscale=0.0, num_nodes=num_nodes)

    fitter = Fitter(dir_name)
    fitter.set_models(model)
    fitter.set_simulations(simulation)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        m, s, chain, truth, weight, old_weight, posterior = fitter.load()
        chain["w"] = old_weight
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior)
        c.configure(color_params="w")
        c.plot(filename=plot_filename, truth=truth, parameters=15)
