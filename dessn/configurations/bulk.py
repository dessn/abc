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
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)

    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            pass

    models = [
        ApproximateModelW(prior=True, statonly=False),
        ApproximateModelW(prior=True, statonly=True)
    ]

    ndes = 204
    nlowz = 128
    simulations = [
        [SNANASimulation(ndes, "DES3YR_DES_BULK_G10_SKEW"), SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_G10_SKEW")],
        [SNANASimulation(ndes, "DES3YR_DES_BULK_C11_SKEW"), SNANASimulation(nlowz, "DES3YR_LOWZ_BULK_C11_SKEW")],
    ]
    fitter = Fitter(dir_name)

    # data = models[0].get_data(simulations[0], 0, plot=False)  # For testing

    fitter.set_models(*models)
    fitter.set_simulations(*simulations)
    ncosmo = 100
    fitter.set_num_cosmologies(ncosmo)
    fitter.set_max_steps(3000)
    fitter.set_num_walkers(2)
    fitter.set_num_cpu(600)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        import numpy as np

        res = fitter.load(split_models=True, split_sims=True, split_cosmo=True, squeeze=False)
        # res2 = fitter.load(split_models=True, split_sims=False)

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
                cc.add_chain(chain, weights=weight, posterior=posterior, name=name, plot_contour=False, plot_point=True,
                           color=col, marker_style=ms, marker_size=msize)
            else:
                c1.add_chain(chain, weights=weight, posterior=posterior, name=name)

        for k in ws.keys():
            print("%s %5.3f %5.3f (%5.3f) : %4.2f" % (k, np.mean(ws[k]), np.mean(ws_std[k]), np.std(ws[k]), np.sqrt(100)*(-1-np.mean(ws[k]))/np.std(ws[k])))
        # c2.configure(spacing=1.0, sigma2d=False, flip=False, shade=True, linestyles=ls, colors=cs, shade_gradient=1.4, shade_alpha=shades, linewidths=1.2)
        # c2.plotter.plot_summary(filename=[pfn + "2.png", pfn + "2.pdf"], parameters=["$w$"], truth=[-1.0], figsize=1.5, errorbar=True)
        # c2.plotter.plot(filename=[pfn + "_small.png", pfn + "_small.pdf"], parameters=2, truth=truth, extents={"$w$": (-1.4, -0.7)}, figsize="column")

        if False:
            c2.configure(global_point=False, plot_hists=False, legend_artists=True)
            c3.configure(global_point=False, plot_hists=False, legend_artists=True)
            ex = {r"\Omega_m$": (0.27, 0.33), "$w$": (-1.35, -0.7), r"$\alpha$": (0.12, 0.18), r"$\beta$": (2.6, 4.)}
            c2.plotter.plot(filename=[pfn + "_points_g10.png", pfn + "_points_g10.pdf"], parameters=4,
                            truth=truth, extents=ex, figsize=1.0, )
            c3.plotter.plot(filename=[pfn + "_points_c11.png", pfn + "_points_c11.pdf"], parameters=4,
                            truth=truth, extents=ex, figsize=1.0)

        if True:
            bbcs = [np.loadtxt(plot_dir + "dillon_g10.txt"), np.loadtxt(plot_dir + "dillon_c11.txt")]

            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=2, figsize=(5, 9))
            for t, ax, bbc in zip(["G10", "C11"], axes, bbcs):
                for key in ws.keys():
                    if t not in key:
                        continue
                    means_steve = np.array(ws[key])
                    std_steve = np.array(ws_std[key])

                    means_bbc = bbc[:, 0] - 1
                    std_bbc = bbc[:, 1]

                    print(means_steve.shape, means_bbc.shape)

                    minv = min(np.min(means_steve), np.min(means_bbc)) - 0.02
                    maxv = max(np.max(means_steve), np.max(means_bbc)) + 0.02
                    color = "k" if "Syst" not in key else ("g" if "G10" in key else "r")
                    ecolor = "#888888" if "Syst" not in key else ("#abf4bd" if "G10" in key else "#f4abab")
                    ax.errorbar(means_steve, means_bbc, yerr=std_steve, xerr=std_bbc, errorevery=2,
                                fmt="o", capsize=2, markersize=5, c=color, ecolor=ecolor, elinewidth=0.5)

                ax.plot([minv, maxv], [minv, maxv], c='k', linewidth=1)
                ax.set_xlim([minv, maxv])
                ax.set_ylim([minv, maxv])
                ax.set_xlabel("BBC $w$")
                ax.set_ylabel("Steve $w$")
            # plt.subplots_adjust(wspace=0, hspace=0)

            fig.tight_layout()
            plt.show()




        # c2.plotter.plot(filename=pfn + "_big.png", parameters=14, truth=truth)
        # c2.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=7)
        # with open(pfn + "_summary.txt", "w") as f:
        #     f.write(c2.analysis.get_latex_table(transpose=True))
        #
        # pps = ['$\\Omega_m$', '$w$', '$\\alpha$', '$\\beta$', '$\\langle M_B \\rangle$',
        #        '$\\sigma_{\\rm m_B}^{0}$', '$\\sigma_{\\rm m_B}^{1}$', '$\\sigma_{x_1}^{0}$',
        #        '$\\sigma_{x_1}^{1}$', '$\\sigma_{c}^{0}$', '$\\sigma_{c}^{1}$', '$\\alpha_c^{0}$',
        #        '$\\alpha_c^{1}$', '$\\kappa_{c0}^{0}$', '$\\kappa_{c0}^{1}$', '$\\kappa_{c1}^{0}$',
        #        '$\\kappa_{c1}^{1}$', '$s_c^{0}$', '$\\delta(0)$', '$\\delta(\\infty)/\\delta(0)$',
        #        '$\\langle x_1^{0} \\rangle$', '$\\langle x_1^{1} \\rangle$', '$\\langle x_1^{2} \\rangle$',
        #        '$\\langle x_1^{3} \\rangle$', '$\\langle x_1^{4} \\rangle$', '$\\langle x_1^{5} \\rangle$',
        #        '$\\langle x_1^{6} \\rangle$', '$\\langle x_1^{7} \\rangle$', '$\\langle c^{0} \\rangle$',
        #        '$\\langle c^{1} \\rangle$', '$\\langle c^{2} \\rangle$', '$\\langle c^{3} \\rangle$',
        #        '$\\langle c^{4} \\rangle$', '$\\langle c^{5} \\rangle$', '$\\langle c^{6} \\rangle$',
        #        '$\\langle c^{7} \\rangle$']
        # p_cor, c_cor = c2.analysis.get_correlations(chain=0, parameters=pps)
        # import numpy as np
        # import matplotlib.pyplot as plt
        # from matplotlib import ticker
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        #
        # fig, ax = plt.subplots(figsize=(9, 9))
        # handle = ax.imshow(c_cor, cmap="magma")
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="3%", pad=0.05)
        # fig.colorbar(handle, cax=cax)
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax.set_yticklabels([''] + pps, fontsize=10)
        # ax.set_xticklabels([''] + pps, fontsize=10)
        # for tick in ax.get_xticklabels():
        #     tick.set_rotation(90)
        # plt.tight_layout()
        # fig.savefig(pfn + "_cor.pdf", bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.05)
        # fig.savefig(pfn + "_cor.png", bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.05)
        #
        # with open(pfn + "_cor_tab.txt", 'w') as f:
        #     f.write(c2.chains[0].name)
        #     f.write(c2.analysis.get_correlation_table(chain=0))
        #     f.write(c2.chains[1].name)
        #     f.write(c2.analysis.get_correlation_table(chain=1))
        #     f.write(c2.chains[2].name)
        #     f.write(c2.analysis.get_correlation_table(chain=2))
        #     f.write(c2.chains[3].name)
        #     f.write(c2.analysis.get_correlation_table(chain=3))

        # print(c2.analysis.get_correlations(chain=1))
        # print(c2.analysis.get_correlations(chain=2))
        # print(c2.analysis.get_correlations(chain=3))
        # c2.plotter.plot(filename=pfn + "_big2.png", parameters=31, truth=truth)

        #
        #
        #
        # KDE Stacking

        # import numpy as np
        #
        # def convert_to_stdev(sigma):  # pragma: no cover
        #     # From astroML
        #     shape = sigma.shape
        #     sigma = sigma.ravel()
        #     i_sort = np.argsort(sigma)[::-1]
        #     i_unsort = np.argsort(i_sort)
        #
        #     sigma_cumsum = 1.0 * sigma[i_sort].cumsum()
        #     sigma_cumsum /= sigma_cumsum[-1]
        #
        #     val = sigma_cumsum[i_unsort].reshape(shape)
        #     return val
        #
        # res = fitter.load(split_models=True, split_sims=True, split_cosmo=True, squeeze=False)
        # from fastkde import fastKDE
        # from scipy.interpolate import interp2d
        # theta_prob_c11, theta_prob_g10 = [], []
        # for m, s, ci, chain, truth, weight, old_weight, posterior in res:
        #     sim_name = s[0].simulation_name
        #     if not m.statonly:
        #         continue
        #
        #     omega_m = chain['$\\Omega_m$']
        #     w = chain['$w$']
        #     myPDF, axes = fastKDE.pdf(omega_m, w)
        #     probs = convert_to_stdev(myPDF)
        #
        #     # Get prob value at truth value 0,0
        #     v1, v2 = axes
        #     inter = interp2d(v1, v2, probs)
        #     val = inter(0.3, -1)
        #     if "C11" in sim_name:
        #         theta_prob_c11.append(val[0])
        #     elif "G10" in sim_name:
        #         theta_prob_g10.append(val[0])
        #     else:
        #         print("Oh no ", sim_name)
        #
        # print(theta_prob_g10)
        # m = np.mean(theta_prob_g10)
        # me = np.std(theta_prob_g10) / np.sqrt(len(theta_prob_g10))
        # print(len(theta_prob_g10))
        # print("G10: Mean is %0.3f, error is %0.3f, not including KDE error" % (m, me))
        #
        # print(theta_prob_c11)
        # m = np.mean(theta_prob_c11)
        # me = np.std(theta_prob_c11) / np.sqrt(len(theta_prob_c11))
        # print("C11: Mean is %0.3f, error is %0.3f, not including KDE error" % (m, me))



            # res3 = fitter.load(split_models=True, split_sims=True, split_cosmo=True)
        # wdict = {}
        # for m, s, ci, chain, truth, weight, old_weight, posterior in res3:
        #     if isinstance(m, ApproximateModelW):
        #         sim_name = s[0].simulation_name
        #         if "MAGSMEAR" in sim_name:
        #             name = "Coherent"
        #         elif "G10" in sim_name:
        #             name = "G10"
        #         elif "C11" in sim_name:
        #             name = "C11"
        #         else:
        #             name = sim_name.replace("DES3YR_DES_", "").replace("_", " ").replace("SKEW", "SK16")
        #         name = "%s %s" % (name, "Stat" if m.statonly else "Stat+Syst")
        #         if wdict.get(name) is None:
        #             wdict[name] = []
        #         wdict[name].append([ci, chain])
        # import numpy as np
        # with open(pfn + "_comp.txt", 'w') as f:
        #     f.write("%10s %5s(%5s) %5s %5s\n" % ("Key", "<w>", "<werr>", "std<w>", "bias"))
        #     for key in sorted(wdict.keys()):
        #         ws = [cc[1]["$w$"] for cc in wdict[key]]
        #         indexes = [cc[0] for cc in wdict[key]]
        #         means = [np.mean(w) for w in ws]
        #         stds = [np.std(w) for w in ws]
        #         name2 = pfn + key.replace(" ", "_") + ".txt"
        #         with open(name2, "w") as f2:
        #             for i in range(ncosmo):
        #                 if i in indexes:
        #                     f2.write("%0.5f\n" % means[indexes.index(i)])
        #                 else:
        #                     f2.write("0\n")
        #         mean_mean = np.average(means, weights=1 / (np.array(stds) ** 2))
        #         mean_std = np.mean(stds)
        #         bias = (mean_mean + 1) / mean_std
        #         f.write("%10s %0.4f(%0.4f) %0.4f %0.4f\n" % (key, mean_mean, mean_std, np.std(means), bias))
