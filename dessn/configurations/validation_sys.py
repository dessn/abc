import os
import logging
import socket

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

    model = ApproximateModelW(prior=True, statonly=True, global_calibration=1)
    # Turn off mass and skewness for easy test

    ndes = 600
    nlowz = 300
    simulations = [
            [SNANASimulation(ndes, "DES3Y_DES_VALIDATION_STATONLYsys"), SNANASimulation(nlowz, "DES3Y_LOWZ_VALIDATION_STATONLYsys")]
        ]
    fitter = Fitter(dir_name)

    # data = model.get_data(simulations[0], 0)  # For testing
    # exit()

    fitter.set_models(model)
    fitter.set_simulations(*simulations)
    fitter.set_num_cosmologies(200)
    fitter.set_max_steps(3000)
    fitter.set_num_walkers(2)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        import numpy as np
        res = fitter.load(split_cosmo=True, split_sims=True)

        ws = {}
        for m, s, chain, truth, weight, old_weight, posterior in res:
            key = s[0].simulation_name
            if key not in ws:
                ws[key] = []
            ws[key].append([chain["$w$"].mean(), np.std(chain["$w$"])])

        # for key in ws.keys():
        #     vals = np.array(ws[key])
            # print(key, vals[:, 0])
        for key in sorted(ws.keys()):
            vals = np.array(ws[key])
            print("%35s %8.4f %8.4f %8.4f %8.4f"
                  % (key, np.average(vals[:, 0], weights=1/(vals[:, 1]**2)),
                     np.std(vals[:, 0]), np.std(vals[:, 0])/np.sqrt(100), np.mean(vals[:, 1])))


