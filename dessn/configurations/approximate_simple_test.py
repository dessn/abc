import os
import logging
import socket
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModelFixedW
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
    model = ApproximateModelFixedW(num_nodes=num_nodes, global_calibration=1)
    simulation = [SimpleSimulation(700, alpha_c=0, mass=True, dscale=0.08, num_nodes=num_nodes),
                  SimpleSimulation(300, alpha_c=0, mass=True, dscale=0.08, num_nodes=num_nodes, lowz=True)]

    fitter = Fitter(dir_name)
    fitter.set_models(model)
    fitter.set_simulations(simulation)
    fitter.set_num_cosmologies(25)
    fitter.set_num_walkers(10)
    fitter.set_max_steps(3000)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer

        m, s, chain, truth, weight, old_weight, posterior = fitter.load()

        print("Plotting posteriors")
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="Combined")
        c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, contour_labels="confidence", summary=False,
                    plot_hists=False)
        parameters = [r"$\Omega_m$", r"$\alpha$", r"$\beta$"]
        print(c.analysis.get_latex_table(transpose=True))
        with open(plot_filename + "_cosmo_params.txt", 'w') as f:
            f.write(c.analysis.get_latex_table(parameters=parameters))
        c.plotter.plot(filename=plot_filename + "_cosmo.png", truth=truth, parameters=parameters, figsize="column")
        c.plotter.plot(filename=plot_filename + "_cosmo.pdf", truth=truth, parameters=parameters, figsize="column")

        print("Plotting summaries")
        res = fitter.load(split_cosmo=True)
        for i, (m, s, chain, truth, weight, old_weight, posterior) in enumerate(res):
            c.add_chain(chain, weights=weight, posterior=posterior, name="Realisation %d" % i)
        c.plotter.plot_summary(filename=plot_filename + "_summary.png", truth=truth, parameters=parameters, errorbar=True, extra_parameter_spacing=2.0)
        c.plotter.plot_summary(filename=plot_filename + "_summary_big.png", truth=truth)

        exit()
        print("Plotting distributions")
        c = ChainConsumer()
        c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
        c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False, sigma2d=False,
                    contour_labels="confidence")
        c.plotter.plot_distributions(filename=plot_filename + "_dist.png", truth=truth, col_wrap=6)
        c.plotter.plot_distributions(filename=plot_filename + "_dist.pdf", truth=truth, col_wrap=6)

        print("Saving Parameter values")
        with open(plot_filename + "_all_params.txt", 'w') as f:
            f.write(c.analysis.get_latex_table(transpose=True))

        print("Plotting big triangle. This might take a while")
        c.plotter.plot(filename=plot_filename + "_big.png", truth=truth, parameters=10)
        c.plotter.plot_walks(filename=plot_filename + "_walk.png", truth=truth, parameters=3)
