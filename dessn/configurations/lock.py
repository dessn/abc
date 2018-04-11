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

    models = [
        ApproximateModelW(prior=True, statonly=False),
        ApproximateModelW(prior=True, statonly=True),
        ApproximateModelW(prior=True, statonly=True, lock_systematics=True),
    ]

    ndes = 204
    nlowz = 128
    simulations = [
        [SNANASimulation(ndes, "DES3YR_DES_BULK_G10_SKEW"), SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_G10_SKEW")],
        # [SNANASimulation(ndes, "DES3YR_DES_BULK_C11_SKEW"), SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_C11_SKEW")],
    ]
    fitter = Fitter(dir_name)
    fitter.set_models(*models)
    fitter.set_simulations(*simulations)
    ncosmo = 10
    fitter.set_num_cosmologies(ncosmo)
    fitter.set_max_steps(3000)
    fitter.set_num_walkers(1)
    fitter.set_num_cpu(500)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        res = fitter.load(split_models=True, split_sims=True, squeeze=False)
