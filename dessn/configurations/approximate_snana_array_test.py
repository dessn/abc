import os
import logging
import socket
from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModel
from dessn.framework.simulations.snana import SNANASimulationSkewed0p2, SNANASimulationGauss0p2, \
    SNANASimulationGauss0p3, SNANASimulationGauss0p4

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(os.path.abspath(__file__)) + "/output/" + os.path.basename(__file__)[:-3]
    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/"
    plot_filename = plot_dir + os.path.basename(__file__)[:-3] + ".png"
    file = os.path.abspath(__file__)
    print(dir_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    model = ApproximateModel(500)
    simulations = [SNANASimulationSkewed0p2(), SNANASimulationGauss0p2(), SNANASimulationGauss0p3(), SNANASimulationGauss0p4()]

    fitter = Fitter(dir_name)
    fitter.set_models(model)
    fitter.set_simulations(*simulations)
    fitter.set_num_walkers(5)

    h = socket.gethostname()
    if h != "smp-hk5pn72":  # The hostname of my laptop. Only will work for me, ha!
        fitter.fit(file)
    else:
        val = r"$\Omega_m$"
        nval = r"$\Delta \Omega_m$"

        from chainconsumer import ChainConsumer
        results = fitter.load()
        c = ChainConsumer()
        for i, (m, s, chain, truth, weight, old_weight, posterior) in enumerate(results):
            name = "%s_%s" % (m.get_name(), s.get_name())
            chain[nval] = chain[val] - truth[val]
            truth[nval] = 0.0
            c.add_chain(chain, weights=weight, posterior=posterior, name=name.replace("_", "\_"))

            cc = ChainConsumer()
            cc.add_chain(chain, posterior=posterior, name=name.replace("_", "\_"))
            cc.add_chain(chain, weights=weight, posterior=posterior, name=name.replace("_", "\_") + "stan")
            cc.plot(filename=plot_filename.replace(".png", "%s.png" % name), truth=truth, parameters=9)

        parameters = [nval, '$\\alpha$', '$\\beta$', '$\\langle M_B \\rangle$',
                      '$\\sigma_{\\rm m_B}$', '$\\sigma_{x_1}$', '$\\sigma_c$']
        c.plot(filename=plot_filename, truth=truth, parameters=parameters)
