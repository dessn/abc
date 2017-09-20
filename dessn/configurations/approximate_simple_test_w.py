import os
import logging
import socket
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModel
from dessn.framework.simulations.simple import SimpleSimulation

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    dir_name = plot_dir + "output/"
    plot_filename = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)
    print(dir_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    num_nodes = 4
    model = ApproximateModel(num_nodes=num_nodes, global_calibration=1)
    simulation = [SimpleSimulation(300, alpha_c=0, mass=True, dscale=0.08, num_nodes=num_nodes),
                  SimpleSimulation(200, alpha_c=0, mass=True, dscale=0.08, num_nodes=num_nodes, lowz=True)]
    fitter = Fitter(dir_name)
    fitter.set_models(model)
    fitter.set_simulations(simulation)
    fitter.set_num_cosmologies(225)
    fitter.set_num_walkers(1)
    fitter.set_max_steps(5000)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        m, s, chain, truth, weight, old_weight, posterior = fitter.load()

        print("Plotting posteriors")
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        c.configure(spacing=1.0, diagonal_tick_labels=False)
        parameters = [r"$\Omega_m$", r"$w$"]
        print(c.analysis.get_latex_table(transpose=True))
        with open(plot_filename + "_cosmo_params.txt", 'w') as f:
            f.write(c.analysis.get_latex_table(parameters=parameters))
        c.plotter.plot(filename=plot_filename + "_cosmo.png", truth=truth, parameters=parameters, figsize="column")
        c.plotter.plot(filename=plot_filename + "_cosmo.pdf", truth=truth, parameters=parameters, figsize="column")

        print("Plotting distributions")
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False)
        c.plotter.plot_distributions(filename=plot_filename + "_dist.png", truth=truth, col_wrap=6)
        c.plotter.plot_distributions(filename=plot_filename + "_dist.pdf", truth=truth, col_wrap=6)

        print("Saving Parameter values")
        with open(plot_filename + "_all_params.txt", 'w') as f:
            f.write(c.analysis.get_latex_table(transpose=True))

        print("Plotting big triangle. This might take a while")
        c.plotter.plot(filename=plot_filename + "_big.png", truth=truth, parameters=10)
        c.plotter.plot_walks(filename=plot_filename + "_walk.png", truth=truth, parameters=3)

