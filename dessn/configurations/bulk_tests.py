import os
import logging
import socket

from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModelW, ApproximateModel
from dessn.framework.simulations.snana import SNANASimulation
from dessn.general.helper import weighted_avg_and_std

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
        ApproximateModelW(prior=True, statonly=True)
    ]

    ndes = 204
    nlowz = 128
    sim = False
    disp = False
    simulations = [
        [SNANASimulation(ndes, "DES3YR_DES_SAM_G10_SKEW_MINUIT_v8", use_sim=sim, add_disp=disp), SNANASimulation(nlowz, "DES3YR_LOWZ_SAM_G10_SKEW_MINUIT_v8", use_sim=sim, add_disp=disp)],
        [SNANASimulation(ndes, "DES3YR_DES_SAM_C11_SKEW_MINUIT_v8", use_sim=sim, add_disp=disp), SNANASimulation(nlowz, "DES3YR_LOWZ_SAM_C11_SKEW_MINUIT_v8", use_sim=sim, add_disp=disp)],
        [SNANASimulation(ndes, "DES3YR_DES_SAM_G10_SKEW_NOMAGERR_v8", use_sim=sim, add_disp=disp), SNANASimulation(nlowz, "DES3YR_LOWZ_SAM_G10_SKEW_NOMAGERR_v8", use_sim=sim, add_disp=disp)],
        [SNANASimulation(ndes, "DES3YR_DES_SAM_C11_SKEW_NOMAGERR_v8", use_sim=sim, add_disp=disp), SNANASimulation(nlowz, "DES3YR_LOWZ_SAM_C11_SKEW_NOMAGERR_v8", use_sim=sim, add_disp=disp)],
    ]
    fitter = Fitter(dir_name)

    # data = models[0].get_data(simulations[0], 0, plot=False)  # For testing
    # exit()
    fitter.set_models(*models)
    fitter.set_simulations(*simulations)
    ncosmo = 100
    fitter.set_num_cosmologies(ncosmo)
    fitter.set_max_steps(3000)
    fitter.set_num_walkers(1)
    fitter.set_num_cpu(600)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        import numpy as np

        if True:
            res = fitter.load(split_models=True, split_sims=True, split_cosmo=True, squeeze=False)

            c1, c2, c3 = ChainConsumer(), ChainConsumer(), ChainConsumer()
            ls = []
            shades = []
            cs = ['b', 'g', 'p', 'a']
            cs = ['g', 'r', 'k', 'k']

            ws = {}
            ws_std = {}
            for m, s, ci, chain, truth, weight, old_weight, posterior in res:
                sim_name = s[0].simulation_name
                col = "k"
                ms = "+"
                msize = 25
                if "MAGSMEAR" in sim_name:
                    name = "Coherent"
                elif "G10" in sim_name:
                    name = "G10"
                    col = "g"
                elif "C11" in sim_name:
                    name = "C11"
                    col = "r"
                else:
                    name = sim_name.replace("DES3YR_DES_", "").replace("_", " ").replace("SKEW", "SK16")
                name = "%s %s" % (name, "Stat" if m.statonly else "Stat+Syst")
                # if s[0].bias_cor:
                #     name += " Biascor"
                if m.statonly:
                    ls.append("--")
                    shades.append(0.0)
                    col = "k"
                    ms = "^"
                    msize = 5
                else:
                    ls.append("-")
                    shades.append(0.7)
                if isinstance(m, ApproximateModelW):
                    if "C11" in name:
                        cc = c3
                    else:
                        cc = c2
                    if ws.get(name) is None:
                        ws[name] = []
                    if ws_std.get(name) is None:
                        ws_std[name] = []
                    w_mean, w_std = weighted_avg_and_std(chain["$w$"], weight)
                    ws[name].append(w_mean)
                    ws_std[name].append(w_std)
                    cc.add_chain(chain, weights=weight, posterior=posterior, name=name, plot_contour=False, plot_point=True, color=col, marker_style=ms, marker_size=msize)
                else:
                    c1.add_chain(chain, weights=weight, posterior=posterior, name=name)
            with open(pfn + "res.txt", "w") as f:
                for k in ws.keys():
                    s = "%s %5.3f %5.3f (%5.3f) : %4.2f\n" % (k, np.mean(ws[k]), np.mean(ws_std[k]), np.std(ws[k]), np.sqrt(100)*(-1-np.mean(ws[k]))/np.std(ws[k]))
                    print(s)
                    f.write(s)

            if False:
                c2.configure(spacing=1.0, sigma2d=False, flip=False, shade=True, linestyles=ls, colors=cs, shade_gradient=1.4, shade_alpha=shades, linewidths=1.2)
                c2.plotter.plot_summary(filename=[pfn + "2.png", pfn + "2.pdf"], parameters=["$w$"], truth=[-1.0], figsize=1.5, errorbar=True)
                c2.plotter.plot(filename=[pfn + "_small.png", pfn + "_small.pdf"], parameters=2, truth=truth, extents={"$w$": (-1.4, -0.7)}, figsize="column")

            if False:
                c2.configure(global_point=False, plot_hists=False, legend_artists=True)
                c3.configure(global_point=False, plot_hists=False, legend_artists=True)
                ex = {r"\Omega_m$": (0.27, 0.33), "$w$": (-1.35, -0.7), r"$\alpha$": (0.12, 0.18), r"$\beta$": (2.6, 4.)}
                c2.plotter.plot(filename=[pfn + "_points_g10.png", pfn + "_points_g10.pdf"], parameters=4,
                                truth=truth, extents=ex, figsize=1.0, )
                c3.plotter.plot(filename=[pfn + "_points_c11.png", pfn + "_points_c11.pdf"], parameters=4,
                                truth=truth, extents=ex, figsize=1.0)

            if False:
                bbcs = [np.loadtxt(plot_dir + "dillon_g10.txt"), np.loadtxt(plot_dir + "dillon_c11.txt")]

                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(nrows=2, figsize=(5, 9))
                for t, ax, bbc in zip(["G10", "C11"], axes, bbcs):
                    for key in ws.keys():
                        if t not in key:
                            continue
                        if "Syst" not in key:
                            continue
                        means_steve = np.array(ws[key])
                        std_steve = np.array(ws_std[key])

                        means_bbc = bbc[:, 0] - 1
                        print(t, np.mean(means_bbc))
                        std_bbc = bbc[:, 1]

                        minv = min(np.min(means_steve), np.min(means_bbc)) - 0.02
                        maxv = max(np.max(means_steve), np.max(means_bbc)) + 0.02
                        color = "k" if "Syst" not in key else ("g" if "G10" in key else "r")
                        ecolor = "#888888" if "Syst" not in key else ("#abf4bd" if "G10" in key else "#f4abab")
                        ax.errorbar(means_steve, means_bbc, yerr=std_steve, xerr=std_bbc, errorevery=2,
                                    fmt="o", capsize=2, markersize=5, c=color, ecolor=ecolor, elinewidth=0.5)

                    ax.plot([minv, maxv], [minv, maxv], c='k', linewidth=1)
                    ax.set_xlim([minv, maxv])
                    ax.set_ylim([minv, maxv])
                    ax.set_ylabel("BBC $w$")
                    ax.set_xlabel("Steve $w$")
                # plt.subplots_adjust(wspace=0, hspace=0)

                fig.tight_layout()
                plt.show()

        if True:
            cc = ChainConsumer()
            res2 = fitter.load(split_models=True, split_sims=True, split_cosmo=False)
            for m, s, ci, chain, truth, weight, old_weight, posterior in res2:
                if "G10" in s[0].simulation_name:
                    name = "G10"
                    c = "g"
                elif "C11" in s[0].simulation_name:
                    name = "C11"
                    c = "r"
                if not m.statonly:
                    continue
                cc.add_chain(chain, weights=weight, posterior=posterior, name=name, plot_contour=True, color=c)
            cc.plotter.plot(parameters=18, filename=pfn + "big.png")
