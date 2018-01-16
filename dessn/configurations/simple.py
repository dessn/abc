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

    models = [ApproximateModel(), ApproximateModelOl(), ApproximateModelW(), ApproximateModelW(prior=True)]
    simulation = [SimpleSimulation(100), SimpleSimulation(100, lowz=True)]

    fitter = Fitter(dir_name)
    fitter.set_models(*models)
    fitter.set_simulations(simulation)
    fitter.set_num_cosmologies(2)
    fitter.set_num_walkers(1)
    fitter.set_max_steps(3000)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        parameters = [r"$\Omega_m$", "$w$", r"$\Omega_\Lambda$"]

        from chainconsumer import ChainConsumer

        res = fitter.load(split_models=True, squeeze=False, split_cosmo=False)

        print("Adding data")
        c1 = ChainConsumer()

        for i, (m, s, ci, chain, truth, weight, old_weight, posterior) in enumerate(res):
            c1.add_chain(chain, weights=weight, posterior=posterior)
            print(truth["$w$"])

        c1.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False)

        # print("Saving table")
        # print(c1.analysis.get_latex_table(transpose=True))
        # with open(pfn + "_cosmo_params.txt", 'w') as f:
        #     f.write(c1.analysis.get_latex_table(parameters=parameters))

        print("Plotting cosmology")
        c1.plotter.plot(filename=[pfn + "_cosmo.png", pfn + "_cosmo.pdf"], parameters=parameters)
        #
        #
        # print("Plotting distributions")
        # c1.plotter.plot_distributions(filename=[pfn + "_dist.png", pfn + "_dist.pdf"], truth=truth, col_wrap=6)
        #
        # print("Saving Parameter values")
        # with open(pfn + "_all_params.txt", 'w') as f:
        #     f.write(c1.analysis.get_latex_table(transpose=True))
        #
        # print("Plotting big triangle. This might take a while")
        # c1.plotter.plot(filename=pfn + "_big.png", truth=truth, parameters=10)
        # c1.plotter.plot_walks(filename=pfn + "_walk.png", truth=truth, parameters=3)
