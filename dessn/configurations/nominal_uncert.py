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
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            pass

    models = [
        ApproximateModelW(prior=True, statonly=True, lock_systematics=True, lock_disp=True, lock_pop=True, lock_drift=True, lock_base=True, apply_efficiency=False),
        ApproximateModelW(prior=True, statonly=True, lock_systematics=True, lock_disp=True, lock_pop=True, lock_drift=True, lock_base=True),
        ApproximateModelW(prior=True, statonly=True, lock_systematics=True, lock_disp=True, lock_pop=True, lock_drift=True),
        ApproximateModelW(prior=True, statonly=True, lock_systematics=True, lock_disp=False, lock_pop=True, lock_drift=True),
        ApproximateModelW(prior=True, statonly=True, lock_systematics=True, lock_disp=False, lock_pop=False, lock_drift=True),
        ApproximateModelW(prior=True, statonly=True, lock_systematics=True, lock_disp=False, lock_pop=False),
        ApproximateModelW(prior=True, statonly=True, lock_systematics=False, lock_disp=False, lock_pop=False),
        ApproximateModelW(prior=True, statonly=False, lock_systematics=False, lock_disp=False, lock_pop=False)
    ]
    # Turn off mass and skewness for easy test

    ndes = 204
    nlowz = 138
    simulations = [
            [SNANASimulation(ndes, "DES3YR_DES_NOMINAL"), SNANASimulation(nlowz, "DES3YR_LOWZ_NOMINAL")],
    ]
    fitter = Fitter(dir_name)

    # data = model[1].get_data(simulations[0], 0)  # For testing
    # exit()

    fitter.set_models(*models)
    fitter.set_simulations(*simulations)
    fitter.set_num_cosmologies(10)
    fitter.set_max_steps(3000)
    fitter.set_num_walkers(6)
    fitter.set_num_cpu(500)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        import numpy as np
        res = fitter.load(split_cosmo=True, split_sims=True)

        ws = {}
        ns = {}
        for m, s, ci, chain, truth, weight, old_weight, posterior in res:
            key = ("Statonly " if m.statonly else "Syst ") \
                  + ("locksyst " if m.lock_systematics else "") \
                  + ("lockdisp " if m.lock_disp else "") \
                  + ("lockpop " if m.lock_pop else "") \
                  + ("lockbase " if m.lock_base else "") \
                  + ("lockdrift " if m.lock_drift else "") \
                  + ("noeff " if not m.apply_efficiency else "")
            key2 = (1 if m.statonly else 0) +  \
                  + (1 if m.lock_disp else 0) \
                  + (1 if m.lock_pop else 0) \
                  + (1 if m.lock_base else 0) \
                  + (1 if m.lock_drift else 0) \
                  + (1 if not m.apply_efficiency else 0)
            if key == "Statonly locksyst ":
                continue
            key2 = 6 - key2
            if key2 not in ws:
                ws[key2] = []
            if key2 not in ns:
                ns[key2] = key
            ws[key2].append([chain["$w$"].mean(), np.std(chain["$w$"])])

        # for key in ws.keys():
        #     vals = np.array(ws[key])
            # print(key, vals[:, 0])
        for key in sorted(ws.keys()):
            vals = np.array(ws[key])
            key_text = ns[key]
            mean = np.average(vals[:, 0], weights=1/(vals[:, 1]**2))
            scatter = np.std(vals[:, 0])
            std = np.mean(vals[:, 1])

            if key != 0:
                prev_key = key - 1
                prev_std = np.mean(np.array(ws[prev_key])[:, 1])
                prev_var = prev_std**2
                diff = np.sqrt(std**2 - prev_std**2)
            else:
                diff = std
            print("%65s %8.3f %8.3f" % (key_text, std, diff))



            # from chainconsumer import ChainConsumer

            # chain[r"$\Omega_m$"] = blind_om(chain[r"$\Omega_m$"])
            # chain["$w$"] = blind_w(chain["$w$"])
            #
            # c, c2 = ChainConsumer(), ChainConsumer()
            # c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
            # c2.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
            # c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, plot_hists=False, sigmas=[0, 1, 2], contour_labels="confidence")
            # c2.configure(statistics="mean")
            #
            # parameters = [r"$\Omega_m$", "$w$"]  # r"$\alpha$", r"$\beta$", r"$\langle M_B \rangle$"]
            # print(c.analysis.get_latex_table(transpose=True))
            # c.plotter.plot(filename=pfn + ".png", truth=truth, parameters=parameters, watermark="Blinded", figsize=1.5)
            # print("Plotting distributions")
            # c = ChainConsumer()
            # c.add_chain(chain, weights=weight, posterior=posterior, name="Approx")
            # c.configure(label_font_size=10, tick_font_size=10, diagonal_tick_labels=False)
            # c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=8)


