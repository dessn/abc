import os
import inspect
from scipy.interpolate import interp1d
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal, norm
from numpy.lib.recfunctions import append_fields
import numpy as np
from dessn.models.d_simple_stan.get_cosmologies import get_cosmology_dictionary
from dessn.models.d_simple_stan.run import run, get_mc_simulation_data, init_fn, get_analysis_data
import pandas as pd


def calculate_bias(chain_dictionary, supernovae, cosmologies, return_mbs=False):
    supernovae = supernovae[supernovae[:, 6] > 0.0]
    supernovae = supernovae[supernovae[:, 0] < 10.3]
    masses = np.ones(supernovae.size)
    redshifts = supernovae[:, 0]
    apparents = supernovae[:, 1]
    stretches = supernovae[:, 2]
    colours = supernovae[:, 3]
    smear = supernovae[:, 4]
    apparents -= smear
    # return np.ones(chain_dictionary["weight"].shape)
    existing_prob = norm.logpdf(colours, 0, 0.1) + norm.logpdf(stretches, 0, 1) + norm.logpdf(smear, 0, 0.1)

    weight = []
    for i in range(chain_dictionary["mean_MB"].size):
        om = np.round(chain_dictionary["Om"][i], decimals=3)
        key = "%0.3f" % om
        mus = cosmologies[key](redshifts)

        dscale = chain_dictionary["dscale"][i]
        dratio = chain_dictionary["dratio"][i]
        redshift_pre_comp = 0.9 + np.power(10, 0.95 * redshifts)
        mass_correction = dscale * (1.9 * (1 - dratio) / redshift_pre_comp + dratio)
        mass_correction = 0
        mabs = apparents - mus + chain_dictionary["alpha"][i] * stretches - chain_dictionary["beta"][i] * colours + mass_correction * masses

        mbx1cs = np.vstack((mabs, stretches, colours)).T
        chain_MB = chain_dictionary["mean_MB"][i]
        chain_x1 = chain_dictionary["mean_x1"][i]
        chain_c = chain_dictionary["mean_c"][i]
        chain_sigmas = np.array([chain_dictionary["sigma_MB"][i], chain_dictionary["sigma_x1"][i], chain_dictionary["sigma_c"][i]])
        chain_sigmas_mat = np.dot(chain_sigmas[:, None], chain_sigmas[None, :])
        chain_correlations = np.dot(chain_dictionary["intrinsic_correlation"][i], chain_dictionary["intrinsic_correlation"][i].T)
        chain_pop_cov = chain_correlations * chain_sigmas_mat
        chain_mean = np.array([chain_MB, chain_x1, chain_c])

        chain_prob = multivariate_normal.logpdf(mbx1cs, chain_mean, chain_pop_cov)
        reweight = logsumexp(chain_prob - existing_prob)
        if reweight < 1:
            for key in chain_dictionary.keys():
                print(key, chain_dictionary[key][i])
        weight.append(reweight)

    weights = np.array(weight)
    if return_mbs:
        mean_mb = chain_dictionary["mean_MB"] - chain_dictionary["alpha"] * chain_dictionary["mean_x1"] + \
                  chain_dictionary["beta"] * chain_dictionary["mean_c"]
        return weights, mean_mb
    return weights


def approx_bias():
    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)
    data_dir = os.path.abspath(dir_name + "/../data")
    pickle_file = data_dir + os.sep + "supernovae_passed.npy"
    supernovae = np.load(pickle_file)
    d = get_cosmology_dictionary()
    data = get_analysis_data()
    n = 1000
    init_vals = [init_fn(data=data) for i in range(n)]
    keys = list(init_vals[0].keys())
    chain_dictionary = {}
    for key in keys:
        chain_dictionary[key] = np.array([elem[key] for elem in init_vals])
    weights, mean_abs = calculate_bias(chain_dictionary, supernovae, d, return_mbs=True)
    import matplotlib.pyplot as plt
    plt.scatter(mean_abs, weights)
    plt.show()


def add_weight_to_chain(chain_dictionary, n_sne):
    print(n_sne)
    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)
    data_dir = os.path.abspath(dir_name + "/../data")

    dump_file = os.path.abspath(data_dir + "/SHINTON_SPEC_SALT2.npy")
    supernovae = np.load(dump_file)

    d = get_cosmology_dictionary()
    # import matplotlib.pyplot as plt
    # plt.hist(supernovae['S2mb'], 30)
    # plt.show()
    # exit()
    weights = calculate_bias(chain_dictionary, supernovae, d)
    existing = chain_dictionary["weight"]
    logw = n_sne * weights - existing
    logw -= logw.min()
    weights = np.exp(-logw)
    chain_dictionary["weight"] = weights
    chain_dictionary["old_weight"] = existing
    return chain_dictionary


def get_approximate_mb_correction():
    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)
    data_dir = os.path.abspath(dir_name + "/../data")

    dump_file = os.path.abspath(data_dir + "/SHINTON_SPEC_SALT2.npy")
    d = np.load(dump_file)
    mask = d[:, 6] > 0.0
    mB = d[:, 1]
    x1 = d[:, 2]
    c = d[:, 3]
    mu = d[:, 5]
    smear = d[:, 4]
    mB -= smear
    alpha = 0.14
    beta = 3.1
    MB = mB - mu + alpha * x1 - beta * c
    # import matplotlib.pyplot as plt
    # mask = mask #& (d["Z"] < 0.3)
    # fig, axes = plt.subplots(4)
    # print(MB[mask])
    # axes[0].hist(MB[mask], 50, normed=True, histtype="step")
    # axes[0].hist(MB, 50, normed=True, histtype="step")
    # axes[1].hist(x1[mask], 50, normed=True, histtype="step")
    # axes[1].hist(x1, 50, normed=True, histtype="step")
    # axes[2].hist(c[mask], 50, normed=True, histtype="step")
    # axes[2].hist(c, 50, normed=True, histtype="step")
    # axes[3].hist(mB[mask], 50, normed=True, histtype="step")
    # axes[3].hist(mB, 50, normed=True, histtype="step")
    # plt.show()
    # exit()

    bins = np.linspace(19.5, 25, 40)
    hist_all, bins = np.histogram(mB, bins=bins)
    hist_passed, _ = np.histogram(mB[mask], bins=bins)
    binc = 0.5 * (bins[:-1] + bins[1:])
    ratio = 1.0 * hist_passed / hist_all
    ratio = ratio / ratio.max()

    inter = interp1d(ratio, binc)
    mean = inter(0.5)
    width = 0.5 * (inter(0.16) - inter(0.84))
    width += (alpha * np.std(x1) + beta * np.std(c))
    # import matplotlib.pyplot as plt
    # from scipy.stats import norm
    # plt.plot(binc, ratio)
    # plt.plot(binc, 1-norm.cdf(binc, mean, width + 0.02))
    # plt.show()
    # exit()
    return mean, width + 0.02


if __name__ == "__main__":
    # add_weight_to_chain(None, 212)
    file = os.path.abspath(__file__)
    stan_model = os.path.dirname(file) + "/model.stan"

    mB_mean, mB_width = get_approximate_mb_correction()
    print(mB_mean, mB_width)
    data = {
        "mB_mean": mB_mean,
        "mB_width": mB_width,
        "snana_dummy": True,
        "sim": False
    }
    # d = get_analysis_data(**data)
    # dd = np.array(d["obs_mBx1c"])
    # mb = dd[:, 0]
    # x1 = dd[:, 1]
    # c = dd[:, 2]
    # zs = d["redshifts"]
    # from astropy.cosmology import FlatLambdaCDM
    # cosmo = FlatLambdaCDM(70, 0.3)
    # mu = cosmo.distmod(zs).value
    # alpha = 0.14
    # beta = 3.1
    # MB = mb - mu + alpha * x1 - beta * c
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(4)
    # axes[0].hist(MB, 20, normed=True, histtype="step")
    # axes[1].hist(x1, 20, normed=True, histtype="step")
    # axes[2].hist(c, 20, normed=True, histtype="step")
    # axes[3].hist(mb, 20, normed=True, histtype="step")
    # plt.show()
    # exit()

    print("Running %s" % file)
    run(data, stan_model, file, weight_function=add_weight_to_chain)
