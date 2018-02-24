import numpy as np
import inspect
import os


def get_planck(restricted=True):
    """ Priors from COM_CosmoParams_fullGrid_R2.00\base_w\plikHM_TT_lowTEB\base_w_plikHM_TT_lowTEB"""
    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)
    results = np.load(dir_name + "/planck.npy")
    weights = results[:, 0]
    likelihood = results[:, 1]
    chain = results[:, 2:]
    param_file = dir_name + "/planck.paramnames"
    with open(param_file) as f:
        params = ["$%s$" % l.split("\t")[1][:-1] for l in f]
    if restricted:
        wanted_params = [r"$\Omega_m$", "$w$"]
        chain = chain[:, [params.index(p) for p in wanted_params]]
        params = wanted_params
    return chain, params, weights, likelihood

if __name__ == "__main__":
    chain, params, weights, likelihood = get_planck()

    om = chain[:, params.index(r"$\Omega_m$")]
    w = chain[:, params.index(r"$w$")]

    from chainconsumer import ChainConsumer
    c = ChainConsumer()
    c.add_chain(chain, parameters=params)
    c.plotter.plot(display=True)
    # import matplotlib.pyplot as plt
    # plt.hist2d(om, w, bins=100, weights=weights)
    # plt.show()
