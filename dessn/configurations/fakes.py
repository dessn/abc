import os
import logging
import socket


from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import FakeModel
from dessn.framework.simulations.simple import SimpleSimulation

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)20s()] %(message)s")
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    models = [FakeModel()]

    simulations = [SimpleSimulation(20)]

    fitter = Fitter(dir_name)
    fitter.set_models(*models)
    fitter.set_simulations(*simulations)
    ncosmo = 300
    fitter.set_num_cosmologies(ncosmo)
    fitter.set_max_steps(3000)
    fitter.set_num_walkers(1)
    fitter.set_num_cpu(600)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        res = fitter.load(split_models=True, split_sims=True, squeeze=False)

        c = ChainConsumer()
        names = []
        for m, s, ci, chain, truth, weight, old_weight, posterior in res:
            name = m.__class__.__name__
            names.append(name)
            c.add_chain(chain, weights=weight, posterior=posterior, name=name)

        with open(pfn + "_summary.txt", "w") as f:
            f.write(c.analysis.get_latex_table(parameters=["$w$"]))

        import numpy as np
        res = fitter.load(split_cosmo=True, squeeze=False)
        ws = {}
        wserr = {}
        for m, s, ci, chain, truth, weight, old_weight, posterior in res:
            key = m.__class__.__name__ + "_" + ("statonly_" if m.statonly else "syst_") + ("lock" if m.lock_systematics else "nolock")
            if ws.get(key) is None:
                ws[key] = []
                wserr[key] = []
            ws[key].append(np.mean(chain["$w$"]))
            wserr[key].append(np.std(chain["$w$"]))

        import matplotlib.pyplot as plt
        for k in ws.keys():
            print(k, np.std(ws[k]), np.mean(wserr[k]))
            plt.hist(wserr[k], histtype='step')
        plt.show()

