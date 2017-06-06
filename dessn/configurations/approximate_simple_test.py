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

    num_nodes = 4
    model = ApproximateModel(num_nodes=num_nodes)
    simulation = [SimpleSimulation(300, alpha_c=0, mass=True, dscale=0.08, num_nodes=num_nodes, contamination=0.05),
                  SimpleSimulation(200, alpha_c=0, mass=True, dscale=0.08, num_nodes=num_nodes, lowz=True)]

    fitter = Fitter(dir_name)
    fitter.set_models(model)
    fitter.set_simulations(simulation)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        m, s, chain, truth, weight, old_weight, posterior = fitter.load()
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        c.configure(color_params="weights")

        parameters = [r"$\Omega_m$", r"$\alpha$", r"$\beta$", r"$\langle M_B \rangle$", r"$\langle M_B^2 \rangle$",
                       r"$\sigma_{M_B}^2$", r"$\delta(0)$", r"$\delta(\infty)/\delta(0)$"]
        print(c.get_latex_table(transpose=True))
        c.plotter.plot(filename=plot_filename, truth=truth, parameters=parameters)

        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False)
        c.plotter.plot_distributions(filename=plot_filename.replace(".png", "_dist.png"), truth=truth, col_wrap=6)
        c.plotter.plot_walks(filename=plot_filename.replace(".png", "_walk.png"), truth=truth, parameters=parameters)
