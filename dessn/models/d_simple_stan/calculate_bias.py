import pickle
import os
import numpy as np
from chainconsumer import ChainConsumer
from scipy.stats import multivariate_normal
from astropy.cosmology import FlatwCDM

from dessn.models.d_simple_stan.run_stan import get_analysis_data


def calculate_bias(chain_dictionary, supernovae):

    passed = np.array([s["passed_cut"] for s in supernovae])
    masses = np.array([s["mass"] for s in supernovae])
    redshifts = np.array([s["redshift"] for s in supernovae])
    apparents = np.array([s["mB"] for s in supernovae])
    colours = np.array([s["c"] for s in supernovae])
    stretches = np.array([s["x1"] for s in supernovae])
    existing_prob = np.array([s["prob"] for s in supernovae])

    weight = []

    speed_dict = {}

    for i in range(chain_dictionary["mean_MB"].size):
        om = np.round(chain_dictionary["Om"][i], decimals=3)
        key = "%0.3f" % om
        if speed_dict.get(key) is None:
            cosmology = FlatwCDM(70.0, om)
            mus = cosmology.distmod(redshifts).value
            speed_dict[key] = mus
        else:
            mus = speed_dict[key]

        dscale = chain_dictionary["dscale"][i]
        dratio = chain_dictionary["dratio"][i]
        mass_correction = dscale * (1.9 * (1 - dratio) / redshifts + dratio)
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

        chain_prob = multivariate_normal.pdf(mbx1cs, chain_mean, chain_pop_cov)
        reweight = chain_prob / existing_prob
        weight.append(np.average(passed, weights=reweight))
        print(weight[-1])

    return weight


if __name__ == "__main__":
    dir_name = os.path.dirname(__file__) or "."
    output_dir = os.path.abspath(dir_name + "/output")
    stan_output_dir = os.path.abspath(dir_name + "/stan_output")
    stan_file = stan_output_dir + os.sep + "stan.pkl"
    pickle_file = output_dir + os.sep + "supernovae.pickle"
    with open(pickle_file, 'rb') as pkl:
        supernovae = pickle.load(pkl)
    with open(stan_file, 'rb') as pkl:
        chain_dictionary = pickle.load(pkl)

    weights = calculate_bias(chain_dictionary, supernovae)
    del chain_dictionary["intrinsic_correlation"]
    for key in list(chain_dictionary.keys()):
        if "_" in key:
            chain_dictionary[key.replace("_", "")] = chain_dictionary[key]
            del chain_dictionary[key]
    del chain_dictionary["Posterior"]
    c = ChainConsumer()
    c.add_chain(chain_dictionary, name="Unweighted")
    n_sne = get_analysis_data()["n_sne"]
    c.add_chain(chain_dictionary, posterior=np.log(weights), name="Reweighted")
    c.add_chain(chain_dictionary, posterior=n_sne * np.log(weights), name="Reweighted Mult")
    c.plot(filename="output/comparison.png")