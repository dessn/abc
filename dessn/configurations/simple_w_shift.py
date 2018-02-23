import os
import logging
import socket
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModel, ApproximateModelOl, ApproximateModelW
from dessn.framework.simulations.simple import SimpleSimulation

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)
    print(dir_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    models = [
        ApproximateModelW(prior=True, statonly=True, frac_shift=0),
        ApproximateModelW(prior=True, statonly=True, frac_shift=1)
    ]
    simulations = [
        [SimpleSimulation(1000, alpha_c=0), SimpleSimulation(1000, alpha_c=0, lowz=True)],
        [SimpleSimulation(1000, alpha_c=2), SimpleSimulation(1000, alpha_c=2, lowz=True)]
    ]

    # print(models[0].get_data(simulations[0], 0))
    # exit()

    fitter = Fitter(dir_name)
    fitter.set_models(*models)
    fitter.set_simulations(*simulations)
    ncosmo = 100
    fitter.set_num_cosmologies(ncosmo)
    fitter.set_num_walkers(1)
    fitter.set_max_steps(2000)
    fitter.set_num_cpu(600)

    h = socket.gethostname()
    import sys
    if h != "smp-hk5pn72" and (len(sys.argv) == 1 or sys.argv[1] != "A"):  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        parameters = [r"$\Omega_m$", r"$w$"]

        from chainconsumer import ChainConsumer
        c1, c2 = ChainConsumer(), ChainConsumer()
        res = fitter.load(squeeze=False)

        ls = []
        cs = ["#086ed3", "lg", "p", "p", "p", "brown", "brown", "c", "c","o", "o", "e", "e", "g", "g", ]
        for i, (m, s, ci, chain, truth, weight, old_weight, posterior) in enumerate(res):
            params = list(chain.keys())
            name_skew = "Gaussian" if s[0].alpha_c == 0 else "Skewed"
            name_approx = "unshifted" if m.frac_shift == 0 else "shifted"
            name = "%s population, %s  normal colour approximation" % (name_skew, name_approx)
            if name_skew == "Gaussian" and name_approx == "unshifted":
                continue
            if name_skew == "Gaussian":
                ls.append("--")
            else:
                if name_approx == "shifted":
                    ls.append(":")
                else:
                    ls.append("-")
            c1.add_chain(chain, weights=weight, posterior=posterior, name=name)
        print(params)
        red_params = [p for p in params if "langle x_1" not in p and "Delta" not in p and "delta" not in p and "Omega" not in p]

        red_params = ['$w$', '$\\sigma_{c}^{0}$', '$\\sigma_{c}^{1}$', '$\\alpha_c^{0}$',
                      '$\\alpha_c^{1}$', '$\\langle c^{0} \\rangle$', '$\\langle c^{1} \\rangle$',
                      '$\\langle c^{2} \\rangle$', '$\\langle c^{3} \\rangle$', '$\\langle c^{4} \\rangle$',
                      '$\\langle c^{5} \\rangle$', '$\\langle c^{6} \\rangle$', '$\\langle c^{7} \\rangle$']
        red_params2 = ['$\\alpha$', '$\\beta$', '$\\langle M_B \\rangle$', '$\\sigma_{\\rm m_B}^{0}$',
                       '$\\sigma_{\\rm m_B}^{1}$', '$\\sigma_{x_1}^{0}$', '$\\sigma_{x_1}^{1}$',
                       '$\\kappa_{c0}^{0}$', '$\\kappa_{c0}^{1}$', '$\\kappa_{c1}^{0}$',
                       '$\\kappa_{c1}^{1}$', '$s_c^{0}$']
        c1.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, linestyles=ls,
                     colors=cs, max_ticks=4, linewidths=1.1)

        c1.plotter.plot(filename=pfn + "_small.png", truth=truth, parameters=2)

        print("Plotting distributions")

        for i, p in enumerate([red_params, red_params + red_params2]):
            fig = c1.plotter.plot_distributions(truth=truth, col_wrap=5, parameters=p)
            ax = fig.get_axes()
            ax[3].set_ylim(0, 5)
            ax[4].set_ylim(0, 5)
            filenames = [pfn + "_dist_%d.png" % i, pfn + "_dist_%d.pdf" % i]
            for f in filenames:
                c1.plotter._save_fig(fig, f, 300)

        print("Saving Parameter values")
        with open(pfn + "_all_params.txt", 'w') as f:
            f.write(c1.analysis.get_latex_table(transpose=True))

        # print("Plotting big triangle. This might take a while")
        c1.plotter.plot(filename=pfn + "_big.png", truth=truth, parameters=20)
        # c1.plotter.plot_walks(filename=pfn + "_walk.png", truth=truth, parameters=3)
        # c2.plotter.plot_summary(filename=pfn + "_summary_big.png", truth=truth)
