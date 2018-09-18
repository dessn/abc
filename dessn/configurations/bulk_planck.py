import os
import logging
import socket

from scipy.stats import gaussian_kde

from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModelW
from dessn.framework.simulations.snana import SNANASimulation
from dessn.planck.planck import get_planck

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)20s()] %(message)s")
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    # plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % "bulk_desg10"
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)

    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            pass

    models = [
        ApproximateModelW(prior=False, statonly=False),
    ]

    ndes = -1  # 204
    nlowz = -1  # 128
    simulations = [
        [SNANASimulation(ndes, "DES3YR_DES_BULK_G10_SKEW_v8", type=None), SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_G10_SKEW_v8")],
        [SNANASimulation(ndes, "DES3YR_DES_BULK_C11_SKEW_v8", type=None), SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_C11_SKEW_v8")],
    ]
    fitter = Fitter(dir_name)

    # data = models[0].get_data(simulations[0], 0, plot=False)  # For testing
    # exit()
    fitter.set_models(*models)
    fitter.set_simulations(*simulations)
    ncosmo = 5
    fitter.set_num_cosmologies(ncosmo)
    fitter.set_max_steps(3000)
    fitter.set_num_walkers(20)
    fitter.set_num_cpu(600)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        import numpy as np

        chain_planck, params, weights, likelihood = get_planck()
        kde = gaussian_kde(chain_planck.T)
        print("Trained")

        res = fitter.load(split_models=True, split_sims=True, split_cosmo=True, squeeze=False)

        c = ChainConsumer()
        for i, (m, s, ci, chain, truth, weight, old_weight, posterior) in enumerate(res):
            sim_name = s[0].simulation_name

            name = "G10" if "G10" in sim_name else "C11"
            subset = np.vstack((chain[r"$\Omega_m$"], chain["$w$"]))
            weights = kde.evaluate(subset)
            print("Eval")
            c.add_chain(subset.T, weights=weights, name="%s %d" % (name, i), parameters=params)

        with open(pfn + "_res.txt", "w") as f:
            s = c.analysis.get_latex_table(transpose=True)
            print(s)
            f.write(s)
