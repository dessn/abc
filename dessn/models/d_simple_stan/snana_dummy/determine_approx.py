import numpy as np
import os
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
from scipy.stats import norm
from dessn.models.d_simple_stan.get_cosmologies import get_cosmology_dictionary
from dessn.models.d_simple_stan.load import load_stan_from_folder
from dessn.models.d_simple_stan.run import get_analysis_data
from dessn.models.d_simple_stan.snana_dummy.run import get_approximate_mb_correction


def get_new_cor(chain, mean_add=0, frac_color=1, a_cut=0, b_cut=0, ratio_extra=1.0):

    mB_mean, mB_width = get_approximate_mb_correction(ratio2=ratio_extra)

    data = {
        "mB_mean": mB_mean,
        "mB_width": mB_width,
        "snana_dummy": True,
        "sim": False
    }
    d = get_analysis_data(**data)
    redshifts = d["redshifts"]
    cosmologies = get_cosmology_dictionary()

    weight = []
    for i in range(chain["mean_MB"].size):
        om = np.round(chain["Om"][i], decimals=3)
        key = "%0.3f" % om
        mus = cosmologies[key](redshifts)

        mb = chain["mean_MB"][i] + mus - (chain["alpha"][i] - a_cut) * chain["mean_x1"][i] + (chain["beta"][i] - b_cut) * chain["mean_c"][i]

        # cc = 1 - norm.cdf(mb, mB_mean, mB_width) + 0.001
        cc = 1 - norm.cdf(mb, mB_mean + mean_add, mB_width)
        w = np.sum(0.01 + np.log(cc) - frac_color * chain["mean_c"][i])
        weight.append(w)
    return np.array(weight)


def get_val(x, full_log_correction, chain):
    if np.abs(x[0]) > 5 or np.abs(x[1]) > 3 or np.abs(x[2]) > 0.2 or np.abs(x[3]) > 4 or np.abs(x[4]) > 2:
        return -np.inf
    w = get_new_cor(chain, mean_add=x[0], frac_color=x[1], a_cut=x[2], b_cut=x[3], ratio_extra=x[4])
    val = -np.std(full_log_correction - w)
    if np.isnan(val):
        val = -np.inf
    return val


if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"
    res = load_stan_from_folder(std, replace=False, merge=True, cut=False)
    chain, posterior, t, p, f, l, w, ow = res

    logw = np.log(w)
    full_log_correction = logw + ow

    import emcee

    ndim, nwalkers = 5, 10
    p0 = [np.random.rand(ndim) for i in range(nwalkers)]

    if False:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, get_val, args=[full_log_correction, chain])
        sampler.run_mcmc(p0, 500)

        schain = sampler.chain[:, 10:, :].reshape((-1, ndim))
        spost = sampler.lnprobability[:, 10:].reshape((-1, 1))
        print(schain.shape, spost.shape)
        tosave = np.hstack((schain, spost))
        print(tosave.shape)
        np.save("schain.npy", tosave)
    else:
        schain = np.load("schain.npy")
        spost = schain[:, -1]
        schain = schain[:, :-1]

    ii = np.argsort(spost)
    schain = schain[ii]
    spost = spost[ii]

    mask = schain[:, 2] < 0.25

    c = ChainConsumer()
    c.add_chain(schain[mask], posterior=spost[mask], parameters=["meanadd", "fraccolor", "acut", "bcut", "ratio"])
    c.plot_walks(filename="schain_walk.png")
    c.plot(filename="schain_plot.png")
    # new_cor = get_new_cor(chain, mean_add=1.5, sigma_add=-1.0)
    # diff = full_log_correction - new_cor
    # print(np.std(diff))
    # diff2 = full_log_correction - ow
    # diff3 = full_log_correction
    #
    # diff -= diff.mean()
    # diff2 -= diff2.mean()
    # diff3 -= diff3.mean()
    #
    # plt.hist(diff, 100, histtype="step", label="new")
    # plt.hist(diff2, 100, histtype="step", label="Original")
    # plt.hist(diff3, 100, histtype="step", label="No approx")
    # plt.legend()
    # plt.show()