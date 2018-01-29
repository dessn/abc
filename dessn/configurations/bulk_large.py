import os
import logging
import socket

from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModelW, ApproximateModel
from dessn.framework.simulations.snana import SNANASimulation


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)20s()] %(message)s")
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # models = [ApproximateModelW(prior=True), ApproximateModelW(prior=True, statonly=True)]
    models = [ApproximateModelW(prior=True, statonly=True)]

    ndes = 500
    nlowz = 250
    fitter = Fitter(dir_name)
    simulations = [
        [SNANASimulation(ndes, "DES3YR_DES_SAMTEST_MAGSMEAR", kappa=0), SNANASimulation(nlowz, "DES3YR_LOWZ_SAMTEST_MAGSMEAR", kappa=0)],
        [SNANASimulation(ndes, "DES3YR_DES_BULK_G10_SKEW", kappa=0), SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_G10_SKEW", kappa=0)],
        [SNANASimulation(ndes, "DES3YR_DES_BULK_C11_SKEW", kappa=0), SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_C11_SKEW", kappa=0)],
        [SNANASimulation(ndes, "DES3YR_DES_SAMTEST_MAGSMEAR", kappa=-3.3), SNANASimulation(nlowz, "DES3YR_LOWZ_SAMTEST_MAGSMEAR", kappa=-3.3)],
        [SNANASimulation(ndes, "DES3YR_DES_BULK_G10_SKEW", kappa=-3.3), SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_G10_SKEW", kappa=-3.3)],
        [SNANASimulation(ndes, "DES3YR_DES_BULK_C11_SKEW", kappa=-3.3), SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_C11_SKEW", kappa=-3.3)],
    ]

    # data = models[0].get_data(simulations[0], 0, plot=True)  # For testing
    # exit()

    fitter.set_models(*models)
    fitter.set_simulations(*simulations)
    ncosmo = 100
    fitter.set_num_cosmologies(ncosmo)
    fitter.set_max_steps(2000)
    fitter.set_num_walkers(1)
    fitter.set_num_cpu(500)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        res = fitter.load(split_models=True, split_sims=True, squeeze=False)
        # res2 = fitter.load(split_models=True, split_sims=False)

        c2 = ChainConsumer()
        ls = ['-', '-', '-', '--', '--', '--']
        cs = ['r', 'g', 'b', 'r', 'g', 'b']
        for m, s, ci, chain, truth, weight, old_weight, posterior in res:
            name = s[0].simulation_name.replace("DES3YR_DES_", "").replace("_", " ").replace("SKEW", "SK16")\
                .replace("SAMTEST", "").replace("BULK", "")
            name = "%s %0.1f" % (name, s[0].kappa)
            # if s[0].kappa != 0:
            #     continue
            c2.add_chain(chain, weights=weight, posterior=posterior, name=name)

        c2.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, flip=False, shade=True,
                     linestyles=ls, colors=cs)
        c2.plotter.plot_summary(filename=pfn + "2.png", parameters=["$w$"], truth=[-1.0], figsize=1.5, errorbar=True)
        c2.plotter.plot(filename=pfn + "_small.png", parameters=2, truth=truth, figsize=2.0, extents={"$w$": (-1.3, -0.7)})
        c2.plotter.plot(filename=pfn + "_big.png", parameters=5, truth=truth)
        c2.plotter.plot(filename=pfn + "_big2.png", parameters=10, truth=truth)
        # c2.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=7)
        # c2.plotter.plot(filename=pfn + "_big3.png", parameters=31, truth=truth)


