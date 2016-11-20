import os
import inspect
from scipy.interpolate import interp1d
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal
import numpy as np
from dessn.models.d_simple_stan.get_cosmologies import get_cosmology_dictionary
from dessn.models.d_simple_stan.run import run, get_mc_simulation_data


def calculate_bias(chain_dictionary, supernovae, cosmologies):

    mask = supernovae[:, 6] == 1
    supernovae = supernovae[mask, :]
    supernovae = supernovae[:15000, :]
    masses = supernovae[:, 4]
    redshifts = supernovae[:, 5]
    apparents = supernovae[:, 1]
    colours = supernovae[:, 3]
    stretches = supernovae[:, 2]
    existing_prob = supernovae[:, 7]

    weight = []

    for i in range(chain_dictionary["mean_MB"].size):
        om = np.round(chain_dictionary["Om"][i], decimals=3)
        key = "%0.3f" % om
        mus = cosmologies[key](redshifts)

        dscale = chain_dictionary["dscale"][i]
        dratio = chain_dictionary["dratio"][i]
        redshift_pre_comp = 0.9 + np.power(10, 0.95 * redshifts)
        mass_correction = dscale * (1.9 * (1 - dratio) / redshift_pre_comp + dratio)
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
        weight.append(reweight)

    weights = np.array(weight)
    return weights


def add_weight_to_chain(chain_dictionary, n_sne):
    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)
    output_dir = os.path.abspath(dir_name + "/../output")
    pickle_file = output_dir + os.sep + "supernovae_passed.npy"
    supernovae = np.load(pickle_file)
    d = get_cosmology_dictionary()

    weights = calculate_bias(chain_dictionary, supernovae, d)
    existing = chain_dictionary["weight"]

    logw = n_sne * weights - existing
    logw -= logw.min()
    weights = np.exp(-logw)
    chain_dictionary["weight"] = weights
    chain_dictionary["old_weight"] = existing
    return chain_dictionary


def get_approximate_mb_correction():
    d = get_mc_simulation_data()
    mask = d["sim_passed"] == 1
    mB = d["sim_mB"]
    c = d["sim_c"]
    x1 = d["sim_x1"]
    alpha = 0.15
    beta = 4.0

    hist_all, bins = np.histogram(mB, bins=200)
    hist_passed, _ = np.histogram(mB[mask], bins=bins)
    binc = 0.5 * (bins[:-1] + bins[1:])
    ratio = 1.0 * hist_passed / hist_all
    inter = interp1d(ratio, binc)
    mean = inter(0.5)
    width = 0.5 * (inter(0.16) - inter(0.84))
    width += alpha * np.std(x1) + beta * np.std(c)
    return mean, width + 0.02


if __name__ == "__main__":

    file = os.path.abspath(__file__)
    stan_model = os.path.dirname(file) + "/model.stan"

    mB_mean, mB_width = get_approximate_mb_correction()
    print(mB_mean, mB_width)

    data = {
        "mB_mean": mB_mean,
        "mB_width": mB_width
    }
    print("Running %s" % file)
    run(data, stan_model, file, weight_function=add_weight_to_chain)
