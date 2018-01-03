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

    models = [
        ApproximateModelW(prior=True, frac_mean=0.00),
        ApproximateModelW(prior=True, frac_mean=0.50),
        ApproximateModelW(prior=True, frac_mean=1.00),
        ApproximateModelW(prior=True, frac_mean=-0.5),
        ApproximateModelW(prior=True, frac_mean=-1.0)
    ]
    simulation = [SimpleSimulation(600, alpha_c=2, mass=True, dscale=0.08),
                  SimpleSimulation(400, alpha_c=3, mass=True, dscale=0.08, lowz=True)]

    # models[0].get_data(simulation, 0, plot=True)
    # print(models[0].get_data(simulation, 0))
    # exit()

    fitter = Fitter(dir_name)
    fitter.set_models(*models)
    fitter.set_simulations(simulation)
    ncosmo = 20
    fitter.set_num_cosmologies(ncosmo)
    fitter.set_num_walkers(1)
    fitter.set_max_steps(2000)
    fitter.set_num_cpu(500)

    h = socket.gethostname()
    import sys
    if h != "smp-hk5pn72" and (len(sys.argv) == 1 or sys.argv[1] != "A"):  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        parameters = [r"$\Omega_m$", r"$w$"]

        from chainconsumer import ChainConsumer
        c1, c2 = ChainConsumer(), ChainConsumer()
        print("Loading data")
        res = fitter.load()

        print("Adding chains")
        for m, s, ci, chain, truth, weight, old_weight, posterior in res:
            name = "%0.1f %0.1f" % (m.frac_mean, m.frac_sigma)
            c1.add_chain(chain, weights=weight, posterior=posterior, name=name)
            # c2.add_chain(chain, weights=weight, posterior=posterior, name="Combined")
        c1.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, shade=True, shade_alpha=0.3)

        # print("Adding individual realisations")
        # res = fitter.load(split_cosmo=True)
        # for i, (m, s, ci, chain, truth, weight, old_weight, posterior) in enumerate(res):
        #     c2.add_chain(chain, weights=weight, posterior=posterior, name="Realisation %d" % i)
        #
        # c2.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, statistics="mean")
        #
        # import numpy as np
        # ws = [chain.chain[:, chain.parameters.index("$w$")] for chain in c2.chains]
        # means = [np.mean(w) for w in ws]
        # stds = [np.std(w) for w in ws]
        # mean_mean = np.average(means, weights=1 / (np.array(stds) ** 2))
        # mean_std = np.mean(stds)
        # bias = (mean_mean + 1) / mean_std
        #
        # save = "%10s %0.4f(%0.4f) %0.4f %0.4f\n" % ("Simple", mean_mean, mean_std, np.std(means), bias)
        # print(save)
        # with open(pfn + "_test.txt", "w") as f:
        #     f.write(save)

        # print("Saving table")
        # print(c1.analysis.get_latex_table(transpose=True))
        # with open(pfn + "_cosmo_params.txt", 'w') as f:
        #     f.write(c1.analysis.get_latex_table(parameters=parameters))
        #
        print("Plotting cosmology")
        c1.plotter.plot(filename=[pfn + "_cosmo.png", pfn + "_cosmo.pdf"], truth=truth, parameters=parameters, figsize="column")

        print("Plotting summaries")
        c1.plotter.plot_summary(filename=[pfn + "_summary.png", pfn + "_summary.pdf"], truth=truth, parameters=parameters, errorbar=True, extra_parameter_spacing=1.0)

        print("Plotting distributions")
        c1.plotter.plot_distributions(filename=[pfn + "_dist.png", pfn + "_dist.pdf"], truth=truth, col_wrap=6)

        print("Saving Parameter values")
        with open(pfn + "_all_params.txt", 'w') as f:
            f.write(c1.analysis.get_latex_table(transpose=True))

        print("Plotting big triangle. This might take a while")
        c1.plotter.plot(filename=pfn + "_big.png", truth=truth, parameters=10)
        # c1.plotter.plot_walks(filename=pfn + "_walk.png", truth=truth, parameters=3)
        # c2.plotter.plot_summary(filename=pfn + "_summary_big.png", truth=truth)
