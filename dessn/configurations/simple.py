import os
import logging
import socket
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModel, ApproximateModelOl, ApproximateModelW
from dessn.framework.simulations.simple import SimpleSimulation
import matplotlib.pyplot as plt

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
    simulation = [SimpleSimulation(1000), SimpleSimulation(1000, lowz=True)]

    fitter = Fitter(dir_name)
    fitter.set_models(*models)
    fitter.set_simulations(simulation)
    fitter.set_num_cosmologies(100)
    fitter.set_num_walkers(1)
    fitter.set_max_steps(3000)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        parameters = [r"$\Omega_m$", "$w$"]

        from chainconsumer import ChainConsumer

        res = fitter.load(split_models=True, squeeze=False, split_cosmo=False)
        res2 = fitter.load(split_models=True, squeeze=False, split_cosmo=True)

        print("Adding data")
        cb = ChainConsumer()
        names = [r"Flat $\Lambda$CDM", r"$\Lambda$CDM", "Flat $w$CDM", r"Flat $w$CDM + $\Omega_m$ prior"]
        for i, (m, s, ci, chain, truth, weight, old_weight, posterior) in enumerate(res):
            # c = ChainConsumer()
            # c.add_chain(chain, weights=weight, posterior=posterior, name=names[i])
            # if i in [0, 3]:
            #     cb.add_chain(chain, weights=weight, posterior=posterior, name=names[i])
            #
            ps = m.get_cosmo_params()
            n = m.__class__.__name__
            if m.prior:
                n += "OmPrior"

            if len(ps) == 2:
                c2 = ChainConsumer()
                i = 0
                for m2, s2, ci, chain, _, weight, _, pp in res2:
                    if m2 == m and s2 == s:
                        c2.add_chain(chain, weights=weight, posterior=pp, plot_point=True, plot_contour=False, color='k',
                                     marker_style="+", marker_size=20, name="Posterior Maximums")
                        if i == 0:
                            c2.add_chain(chain, weights=weight, plot_point=True, plot_contour=True, color='b',
                                         name="Representative Surface", shade=True, shade_alpha=0.5, kde=1.0, bar_shade=True)
                        i += 1

                c2.plotter.plot(filename=[pfn + n + "2.png"], truth=truth, figsize="column", parameters=ps)

            # c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, sigmas=[0, 1, 2], kde=[False, False, 1.0])
            # c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, sigmas=[0, 0.3, 1, 2], contour_labels="confidence")
            # c.plotter.plot(filename=[pfn + n + ".png", pfn + n + ".pdf"], parameters=ps, truth=truth, figsize="column")
        #
        # print("Saving table")
        # cb.configure(smooth=5, bins=0.6)
        # print(cb.analysis.get_latex_table(transpose=True))
        # with open(pfn + "_cosmo_params.txt", 'w') as f:
        #     f.write(cb.analysis.get_latex_table(parameters=parameters))
