import os
import logging
import socket
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModel, ApproximateModelOl, ApproximateModelW
from dessn.framework.simulations.simple import SimpleSimulation

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)
    print(dir_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    num_nodes = 4
    models = [ApproximateModel(num_nodes=num_nodes),
              ApproximateModelOl(num_nodes=num_nodes),
              ApproximateModelW(num_nodes=num_nodes)]
    simulation = [SimpleSimulation(300, alpha_c=0, mass=True, dscale=0.08, num_nodes=num_nodes),
                  SimpleSimulation(200, alpha_c=0, mass=True, dscale=0.08, num_nodes=num_nodes, lowz=True)]

    fitter = Fitter(dir_name)
    fitter.set_models(*models)
    fitter.set_simulations(simulation)
    fitter.set_num_cosmologies(25)
    fitter.set_num_walkers(10)
    fitter.set_max_steps(2000)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        parameters = [r"$\Omega_m$", r"$\alpha$", r"$\beta$"]

        from chainconsumer import ChainConsumer
        m, s, chain, truth, weight, old_weight, posterior = fitter.load()

        print("Adding data")
        c1, c2 = ChainConsumer(), ChainConsumer()
        c1.add_chain(chain, weights=weight, posterior=posterior, name="Combined")
        c2.add_chain(chain, weights=weight, posterior=posterior, name="Combined")

        print("Adding individual realisations")
        res = fitter.load(split_cosmo=True)
        for i, (m, s, chain, truth, weight, old_weight, posterior) in enumerate(res):
            c2.add_chain(chain, weights=weight, posterior=posterior, name="Realisation %d" % i)

        c1.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, contour_labels="confidence")
        c2.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, contour_labels="confidence")

        print("Saving table")
        print(c1.analysis.get_latex_table(transpose=True))
        with open(pfn + "_cosmo_params.txt", 'w') as f:
            f.write(c1.analysis.get_latex_table(parameters=parameters))

        print("Plotting cosmology")
        c1.plotter.plot(filename=[pfn + "_cosmo.png", pfn + "_cosmo.pdf"], truth=truth, parameters=parameters,
                       figsize="column", chains="Combined")

        print("Plotting summaries")
        c2.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], truth=truth, parameters=parameters, errorbar=True,
                                extra_parameter_spacing=1.0)

        print("Plotting distributions")
        c1.plotter.plot_distributions(filename=[pfn + "_dist.png", pfn + "_dist.pdf"], truth=truth, col_wrap=6)

        print("Saving Parameter values")
        with open(pfn + "_all_params.txt", 'w') as f:
            f.write(c1.analysis.get_latex_table(transpose=True))

        print("Plotting big triangle. This might take a while")
        c1.plotter.plot(filename=pfn + "_big.png", truth=truth, parameters=10)
        c1.plotter.plot_walks(filename=pfn + "_walk.png", truth=truth, parameters=3)
        c2.plotter.plot_summary(filename=pfn + "_summary_big.png", truth=truth)
