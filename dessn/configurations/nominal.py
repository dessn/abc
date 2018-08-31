import os
import logging
import socket
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModelW
from dessn.framework.simulations.snana import SNANASimulation

import numpy as np

def getRMSErr(wvec):
    return np.std(wvec)/np.sqrt(len(wvec))

def getweightedAvg(wvec,werrvec):
    return np.average(wvec,weights=1./np.array(werrvec,dtype='float'))

def getweightedAvgErr(werrvec):
    return np.sqrt(1./np.sum(1./werrvec**2))#https://ned.ipac.caltech.edu/level5/Leo/Stats4_5.html


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)20s()] %(message)s")
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    dir_name = plot_dir + "output/"
    pfn = plot_dir + os.path.basename(__file__)[:-3]

    file = os.path.abspath(__file__)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    models = [ApproximateModelW(prior=True), ApproximateModelW(prior=True, statonly=True)]
    # Turn off mass and skewness for easy test
    ndes = 204
    nlowz = 138
    simulations = [
        [SNANASimulation(ndes, "DES3YR_DES_NOMINAL", type="G10"), SNANASimulation(nlowz, "DES3YR_LOWZ_NOMINAL", type="G10")],
        [SNANASimulation(ndes, "DES3YR_DES_NOMINAL", type="C11"), SNANASimulation(nlowz, "DES3YR_LOWZ_NOMINAL", type="C11")]
    ]

    fitter = Fitter(dir_name)

    # Test systematics
    # data = models[0].get_data(simulation, 0)  # For testing
    # cal = data["deta_dcalib"]
    # print(cal.shape)
    # print(np.max(cal[:, 0, :]))
    # import matplotlib.pyplot as plt
    # plt.imshow(cal[:, 0, :])
    # cbar = plt.colorbar()
    # plt.show()
    # exit()

    fitter.set_models(*models)
    fitter.set_simulations(*simulations)
    fitter.set_num_cosmologies(10)
    fitter.set_max_steps(3000)
    fitter.set_num_walkers(5)
    fitter.set_num_cpu(500)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        from chainconsumer import ChainConsumer
        res = fitter.load()
        c = ChainConsumer()

        for m, s, ci, chain, truth, weight, old_weight, posterior in res:
            name = "Stat + Syst" if not m.statonly else "Stat"
            c.add_chain(chain, weights=weight, posterior=posterior, name=name)

        c.configure(spacing=1.0, diagonal_tick_labels=False, sigma2d=False, plot_hists=False,
                    sigmas=[0, 1, 2], linestyles=["-", "--"], colors=["b", "k"], shade_alpha=[1.0, 0.0])
        parameters = [r"$\Omega_m$", "$w$"]  # r"$\alpha$", r"$\beta$", r"$\langle M_B \rangle$"]
        print(c.analysis.get_latex_table(transpose=True))
        c.plotter.plot(filename=pfn + ".png", parameters=parameters, figsize=1.5)

        res = fitter.load(split_cosmo=True)
        import numpy as np
        ps = [r"$\Omega_m$", "$w$", r"$\alpha$", r"$\beta$", r"$\delta(0)$",
              r"$\sigma_{\rm m_B}^{0}$", r"$\sigma_{\rm m_B}^{1}$",]
        for p in ps:
            mus, stds = [], []
            for m, s, ci, chain, truth, weight, old_weight, posterior in res:
                w = chain[p]
                mus.append(np.mean(w))
                stds.append(np.std(w))
            mus = np.array(mus)
            stds = np.array(stds)
            n = mus.size

            # err(Mean) = RMS(w)/sqrt(n) +- RMS/sqtr(2*n^2)
            # <werr>    = average werr over all sims +- RMS among werr values
            # I am so confused
            wmean = np.average(mus, weights=1/stds)
            wmean_error_from_rms = np.std(mus) / np.sqrt(n)
            wmean_error_on_error = wmean_error_from_rms / np.sqrt(2 * n)
            std = np.sqrt(1 / np.sum(1 / stds**2))
            std_std = np.std(stds)
            print("%s %8.3f %8.3f %8.3f %8.3f %8.3f" % (p, wmean, wmean_error_from_rms, wmean_error_on_error, std, std_std))
            print("%s %6.3f %6.3f %6.3f" % (p, getweightedAvg(mus, stds), getRMSErr(mus), getweightedAvgErr(stds)))
            # print(mus)