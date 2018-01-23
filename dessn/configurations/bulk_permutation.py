import os
import logging
import socket

from dessn.blind.blind import blind_om, blind_w
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModelW
from dessn.framework.simulations.snana import SNANASimulation

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)20s()] %(message)s")
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    model = ApproximateModelW(prior=True, statonly=True)
    # Turn off mass and skewness for easy test

    ndes = 204
    nlowz = 137
    import numpy as np

    simulations = [
        [SNANASimulation(ndes, "DES3YR_DES_BULK_G10_SKEW", shift=np.array([0, 0, 0, 0])) ],
        [SNANASimulation(ndes, "DES3YR_DES_BULK_G10_SKEW", shift=np.array([0.1, 0, 0, 0])) ],
        [SNANASimulation(ndes, "DES3YR_DES_BULK_G10_SKEW", shift=np.array([-0.1, 0, 0, 0])) ],
        [SNANASimulation(ndes, "DES3YR_DES_BULK_G10_SKEW", shift=np.array([0, 0.1, 0, 0])) ],
        [SNANASimulation(ndes, "DES3YR_DES_BULK_G10_SKEW", shift=np.array([0, -0.1, 0, 0])) ],
         # SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_G10_SKEW", shift=np.array([0, 0, 0, 0]))],

        # [SNANASimulation(ndes, "DES3YR_DES_BULK_G10_SKEW", shift=np.array([0.0, 0.1, 0, 0])),
        #  SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_G10_SKEW", shift=np.array([0.0, 0.0, 0, 0]))],
        #
        # [SNANASimulation(ndes, "DES3YR_DES_BULK_G10_SKEW", shift=np.array([0, 0.2, 0, 0])),
        #  SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_G10_SKEW", shift=np.array([0.0, 0, 0, 0]))],
        #
        # [SNANASimulation(ndes, "DES3YR_DES_BULK_G10_SKEW", shift=np.array([0.0, 0.2, 0, 0])),
        #  SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_G10_SKEW", shift=np.array([0.0, 0.2, 0, 0]))],

    ]
    fitter = Fitter(dir_name)

    # data = model.get_data(simulations[0], 0)  # For testing
    # exit()

    fitter.set_models(model)
    fitter.set_simulations(*simulations)
    fitter.set_num_cosmologies(100)
    fitter.set_max_steps(3000)
    fitter.set_num_walkers(1)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        import numpy as np
        from chainconsumer import ChainConsumer

        res = fitter.load(split_models=True, split_sims=True, squeeze=False)

        c = ChainConsumer()

        for m, s, ci, chain, truth, weight, old_weight, posterior in res:
            name = s[0].simulation_name.replace("DES3YR_DES_BULK_", "").replace("_", " ").replace("SKEW", "SK16")
            name = "%s" % (s[0].shift)
            # name = "%s - %s" % (s[0].shift, s[1].shift)
            c.add_chain(chain, weights=weight, posterior=posterior, name=name)

        c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, shade=True)
        c.plotter.plot_summary(filename=pfn + "2.png", parameters=["$w$"], truth=[-1.0], figsize=1.5, errorbar=True)
        c.plotter.plot(filename=pfn + "_big.png", parameters=10, truth=truth)
        c.plotter.plot(filename=pfn + "_big2.png", parameters=31, truth=truth)
        c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=7)

        # res = fitter.load(split_cosmo=True, split_sims=True, squeeze=False)
        #
        # ws = {}
        # for m, s, chain, truth, weight, old_weight, posterior in res:
        #     key = "%s - %s" % (s[0].shift, s[1].shift)
        #     if key not in ws:
        #         ws[key] = []
        #     ws[key].append([chain["$w$"].mean(), np.std(chain["$w$"])])
        #
        #     # for key in ws.keys():
        #     #     vals = np.array(ws[key])
        #     # print(key, vals[:, 0])
        # for i, key in enumerate(sorted(ws.keys())):
        #     vals = np.array(ws[key])
        #     # print("%35s %8.4f %8.4f %8.4f %8.4f"
        #     print("%d %8.4f %8.4f %8.4f %8.4f   %s"
        #           % (i, np.average(vals[:, 0], weights=1 / (vals[:, 1] ** 2)),
        #              np.std(vals[:, 0]), np.std(vals[:, 0]) / np.sqrt(100), np.mean(vals[:, 1]),
        #              key))
