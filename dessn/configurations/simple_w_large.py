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
        ApproximateModelW(prior=True, statonly=True)
    ]
    simulations = [
        [SimpleSimulation(1000), SimpleSimulation(1000, lowz=True)],
        [SimpleSimulation(1000, kappa0=0.04, kappa1=0.04), SimpleSimulation(1000, lowz=True, kappa0=0.04, kappa1=0.04)]
    ]

    # models[0].get_data(simulations[1], 0, plot=True)
    # print(models[0].get_data(simulations[0], 0))
    # exit()

    fitter = Fitter(dir_name)
    fitter.set_models(*models)
    fitter.set_simulations(*simulations)
    ncosmo = 100
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
        res = fitter.load(squeeze=False)

        print("Adding chains")
        ls = []
        cs = ["r", "b", "b", "a", "a", "p", "p", "brown", "brown", "c", "c","o", "o", "e", "e", "g", "g", ]
        for i, (m, s, ci, chain, truth, weight, old_weight, posterior) in enumerate(res):
            name_skew = "Gauss" if s[0].alpha_c == 0 else "Skewed"
            name = "%s Kappas = %0.3f, %0.1f" % (name_skew, s[0].kappa0, s[0].kappa1)
            if name_skew == "Gauss":
                ls.append("--")
            else:
                ls.append("-")
            c1.add_chain(chain, weights=weight, posterior=posterior, name=name)

        c1.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, shade=True, shade_alpha=0.3,
                     linestyles=ls, colors=cs)
        allk = list(chain.keys())
        ps = allk[:5] + [a for a in allk if "sigma" in a] + [a for a in allk if "kappa" in a]

        print("Plotting cosmology")
        c1.plotter.plot(filename=pfn + "_cosmo.png", truth=truth, parameters=parameters, figsize="column")

        print("Plotting distributions")
        c1.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=6)

        print("Plotting big triangle. This might take a while")
        c1.plotter.plot(filename=pfn + "_big.png", truth=truth, parameters=ps)
        # c1.plotter.plot_walks(filename=pfn + "_walk.png", truth=truth, parameters=3)
