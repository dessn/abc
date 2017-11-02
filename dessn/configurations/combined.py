import os
import logging
import socket
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModelW, ApproximateModel, ApproximateModelOl
from dessn.framework.simulations.snana_bulk import SNANACombinedBulk
from dessn.framework.simulations.selection_effects import lowz_sel, des_sel
from dessn.planck.planck import get_planck

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    dir_name = plot_dir + "output/"
    pfn1 = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)
    print(dir_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    models = ApproximateModelW(), ApproximateModelW(systematics_scale=0.001), \
             ApproximateModel(), ApproximateModel(systematics_scale=0.001), \
             ApproximateModelOl(), ApproximateModelOl(systematics_scale=0.001)

    # Turn off mass and skewness for easy test
    simulation = [SNANACombinedBulk(152, ["SHINTON_LOWZ_MATRIX_G10_SKEWC_SKEWX1", "SHINTON_LOWZ_MATRIX_C11_SKEWC_SKEWX1"],
                                    "CombinedLowZ", manual_selection=lowz_sel(), num_calib=50),
                  SNANACombinedBulk(208, ["SHINTON_DES_MATRIX_G10_SKEWC_SKEWX1", "SHINTON_DES_MATRIX_C11_SKEWC_SKEWX1"],
                                    "CombinedDES", manual_selection=des_sel(), num_calib=21)]

    fitter = Fitter(dir_name)
    fitter.set_models(*models)
    fitter.set_simulations(simulation)
    fitter.set_num_cosmologies(100)
    fitter.set_num_walkers(1)
    fitter.set_max_steps(5000)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        results = fitter.load()
        print("Data loaded")

        classes = list(set([r[0].__class__.__name__ for r in results]))
        for cls in classes: # ["ApproximateModelW"]:  #
            pfn = pfn1 + "_" + cls
            c = ChainConsumer()
            parameters = [r"$\Omega_m$"]
            for m, s, chain, truth, weight, old_weight, posterior in results:
                print(m.__class__.__name__)
                if m.__class__.__name__ != cls:
                    continue
                actual_truth = truth  # So sorry about this scope violation
                name = "Statistics" if m.systematics_scale < 0.1 else "Systematics"
                c.add_chain(chain, weights=weight, posterior=posterior, name=name)

            if cls.endswith("W"):
                parameters.append("$w$")

                # So, whatever results I found arent the same as normally used
                p_chain, p_params, p_weight, p_like = get_planck()
                c.add_chain(p_chain, parameters=p_params, weights=p_weight, name="Planck")
                c.configure(linestyles=["-", "--", ":"], colors=["b", "k", "o"], shade_alpha=[1.0, 0.0, 0.3], diagonal_tick_labels=False, kde=[False, False, 1.0])
                extents = {r"$\Omega_m$": [0.1, 0.6], "$w$": [-2, -0.5]}
            else:
                if cls.endswith("Ol"):
                    parameters.append(r"$\Omega_\Lambda$")
                else:
                    parameters.append(r"$\alpha$")
                    parameters.append(r"$\beta$")
                c.configure(linestyles=["-", "--"], colors=["b", "k"], shade_alpha=[1.0, 0.0], diagonal_tick_labels=False)
                extents = None

            print("Latex table for %s" % cls)
            print(c.analysis.get_latex_table(transpose=True))
            with open(pfn + "_summary.txt", 'w') as f:
                f.write(c.analysis.get_latex_table(parameters=parameters))
            fig = c.plotter.plot(filename=[pfn + ".png", pfn + ".pdf"], truth=actual_truth, parameters=parameters, figsize=2.2, extents=extents)
            print("Plotting distributions")
            # c.plotter.plot_distributions(filename=pfn + "_dist.png", truth=truth, col_wrap=8)

