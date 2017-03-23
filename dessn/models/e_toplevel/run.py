import os
import pickle

import shutil
import numpy as np
from numpy.random import uniform
import sys
import socket

from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal, norm
from scipy.misc import logsumexp
from dessn.models.d_simple_stan.get_cosmologies import get_cosmology_dictionary
from dessn.models.d_simple_stan.load_correction_data import load_correction_supernova
from dessn.models.d_simple_stan.load_fitting_data import get_sncosmo_pickle_data, load_fit_snana_correction, \
    get_fitres_data, get_snana_data, load_fit_snana_diff, load_fit_snana_diff2, get_fit_physical_data
from dessn.models.d_simple_stan.run import get_base_data, get_correction_data_from_data_source
from dessn.models.e_toplevel.truth import get_truths_labels_significance


def calculate_bias(chain_dictionary, supernovae, cosmologies, num=None):
    """ Calculates the correct per supernova bias for each step in chain_dictionary,
    using the supernovae sample and dist_mod interpolators found in cosmologies. """
    redshifts = supernovae["redshifts"]
    apparents = supernovae["apparents"]
    stretches = supernovae["stretches"]
    colours = supernovae["colours"]
    existing_prob = supernovae["existing_prob"]
    masses = supernovae["masses"]
    if num is not None:
        redshifts = redshifts[:num]
        apparents = apparents[:num]
        stretches = stretches[:num]
        colours = colours[:num]
        existing_prob = existing_prob[:num]
        masses = masses[:num]

    weight = []
    for i in range(chain_dictionary["mean_MB"].size):
        om = np.round(chain_dictionary["Om"][i], decimals=3)
        key = "%0.3f" % om
        mus = cosmologies[key](redshifts)

        if "dscale" in chain_dictionary.keys() and "dratio" in chain_dictionary.keys():
            dscale = chain_dictionary["dscale"][i]
            dratio = chain_dictionary["dratio"][i]
            redshift_pre_comp = 0.9 + np.power(10, 0.95 * redshifts)
            mass_correction = dscale * (1.9 * (1 - dratio) / redshift_pre_comp + dratio)
        else:
            mass_correction = 0
        mabs = apparents - mus + chain_dictionary["alpha"][i] * stretches - chain_dictionary["beta"][i] * colours + mass_correction * masses

        mbx1cs = np.vstack((mabs, stretches, colours)).T
        chain_MB = chain_dictionary["mean_MB"][i]
        chain_x1 = chain_dictionary["mean_x1"][i]
        chain_c = chain_dictionary["mean_c"][i]
        try:
            chain_sigmas = np.array([chain_dictionary["sigma_MB"][i], chain_dictionary["sigma_x1"][i], chain_dictionary["sigma_c"][i]])
        except KeyError:
            chain_sigmas = np.array([np.exp(chain_dictionary["log_sigma_MB"][i]), np.exp(chain_dictionary["log_sigma_x1"][i]), np.exp(chain_dictionary["log_sigma_c"][i])])

        chain_sigmas_mat = np.dot(chain_sigmas[:, None], chain_sigmas[None, :])
        chain_correlations = np.dot(chain_dictionary["intrinsic_correlation"][i], chain_dictionary["intrinsic_correlation"][i].T)
        chain_pop_cov = chain_correlations * chain_sigmas_mat
        chain_mean = np.array([chain_MB, chain_x1, chain_c])

        chain_prob = multivariate_normal.logpdf(mbx1cs, chain_mean, chain_pop_cov)
        if "alpha_c" in chain_dictionary.keys():
            alpha_c = chain_dictionary["alpha_c"][i]
            skew_prob = norm.logcdf(alpha_c * (colours - chain_c) / chain_dictionary["sigma_c"][i], 0, 1)
            chain_prob += skew_prob
        reweight = logsumexp(chain_prob - existing_prob)
        weight.append(reweight)

    weights = np.array(weight)
    return weights


def add_weight_to_chain(chain_dictionary, n_sne, correction_source, num=None, trim=False, trim_v=-12, shuffle=False):
    # Load supernova for correction
    supernovae = load_correction_supernova(correction_source=correction_source, shuffle=shuffle, only_passed=True)
    # Load premade cosmology dictionary to speed up dist_mod calculation
    d = get_cosmology_dictionary()
    # Get the weights
    weights = calculate_bias(chain_dictionary, supernovae, d, num=num)
    # Get the approximation correction used in stan
    existing = chain_dictionary["weight"]
    # Reweight the chain to match the difference from actual to approximation correction
    logw = existing - n_sne * weights

    # weight is the reweighted chain. calc_weight is the actual (log) correction for one SN. old_weight is the approx
    chain_dictionary["weight"] = logw
    chain_dictionary["calc_weight"] = weights
    chain_dictionary["old_weight"] = existing

    if trim:
        keep = ((logw - logw.max()) > trim_v) | (np.random.uniform(size=logw.size) > 0.99)
        for key in chain_dictionary:
            chain_dictionary[key] = chain_dictionary[key][keep]

    return chain_dictionary


def get_analysis_data(data_source="snana_dummy", n=500, seed=0, add_sim=0, **extra_args):
    """ Gets the full analysis data. That is, the observational data, and all the
    useful things we pre-calculate and give to stan to speed things up.
    """
    np.random.seed(seed)
    data = get_base_data(data_source, n)

    n_sne = data["n_sne"]

    cors = []
    for c in data["obs_mBx1c_cov"]:
        d = np.sqrt(np.diag(c))
        div = (d[:, None] * d[None, :])
        cor = c / div
        cors.append(cor)

    data["obs_mBx1c_cor"] = cors
    redshifts = data["redshifts"]

    num_nodes = 2

    sorted_zs = np.sort(redshifts)
    indexes = np.arange(num_nodes)
    nodes = np.linspace(sorted_zs[5], sorted_zs[-5], num_nodes)
    interps = interp1d(nodes, indexes, kind='linear', fill_value="extrapolate")(redshifts)
    node_weights = np.array([1 - np.abs(v - indexes) for v in interps])
    node_weights *= (node_weights <= 1) & (node_weights >= 0)
    node_weights = np.abs(node_weights)
    reweight = np.sum(node_weights, axis=1)
    node_weights = (node_weights.T / reweight).T

    n_z = 2000

    sim_data = {}
    dz = redshifts.max() / n_z
    zs = sorted(redshifts.tolist())
    n_sim = 0

    added_zs = [0]
    pz = 0
    for z in zs:
        est_point = int((z - pz) / dz)
        if est_point % 2 == 0:
            est_point += 1
        est_point = max(3, est_point)
        new_points = np.linspace(pz, z, est_point)[1:-1].tolist()
        added_zs += new_points
        pz = z
    n_z = n_sne + n_sim + len(added_zs)
    n_simps = int((n_z + 1) / 2)
    to_sort = [(z, -1, -1) for z in added_zs] + [(z, i, -1) for i, z in enumerate(redshifts)]

    to_sort.sort()
    final_redshifts = np.array([z[0] for z in to_sort])
    sorted_vals = [(z[1], i) for i, z in enumerate(to_sort) if z[1] != -1]
    sorted_vals.sort()
    final = [int(z[1] / 2 + 1) for z in sorted_vals]

    update = {
        "n_z": n_z,
        "n_simps": n_simps,
        "zs": final_redshifts,
        "zspo": 1 + final_redshifts,
        "zsom": (1 + final_redshifts) ** 3,
        "redshift_indexes": final,
        "redshift_pre_comp": 0.9 + np.power(10, 0.95 * redshifts),
        "calib_std": np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1]),
        "num_nodes": num_nodes,
        "node_weights": node_weights
    }

    if extra_args is None:
        extra_args = {}

    obs_data = np.array(data["obs_mBx1c"])
    print("Observed data x1 dispersion is %f, colour dispersion is %f"
          % (np.std(obs_data[:, 1]), np.std(obs_data[:, 2])))

    # If you want python2: data.update(update), return data
    return {**data, **update, **sim_data, **extra_args}


def init_fn(n_sne):
    vals = get_truths_labels_significance()
    randoms = {k[0]: uniform(k[4], k[5]) for k in vals}
    for key in randoms:
        if key.find("sigma") == 0:
            randoms["log_%s" % key] = np.log(randoms[key])

    randoms["deviations"] = np.random.normal(scale=0.2, size=(n_sne, 3))
    chol = [[1.0, 0.0, 0.0],
            [np.random.random() * 0.1 - 0.05, np.random.random() * 0.1 + 0.7, 0.0],
            [np.random.random() * 0.1 - 0.05, np.random.random() * 0.1 - 0.05,
             np.random.random() * 0.1 + 0.7]]
    randoms["intrinsic_correlation"] = chol
    randoms["calibration"] = (np.random.uniform(size=8) - 0.5) * 0.2
    randoms["mean_x1"] = (np.random.uniform(size=2) - 0.5) * 0.2
    randoms["mean_c"] = (np.random.uniform(size=2) - 0.5) * 0.1
    randoms["alpha_c"] = (np.random.uniform(size=2) - 0.5) * 0.2
    return randoms


def run_single_input(data_args, stan_model, stan_dir, i, num_walks_per_cosmology=20, weight_function=None):
    n_cosmology = i // num_walks_per_cosmology
    n_run = i % num_walks_per_cosmology
    run_single(data_args, stan_model, stan_dir, n_cosmology, n_run, weight_function=weight_function)


def run_single(data_args, stan_model, stan_dir, n_cosmology, n_run, chains=1, weight_function=None, short=False):
    if short:
        w, n = 500, 1000
    else:
        w, n = 1000, 3000
    data = get_analysis_data(seed=n_cosmology, **data_args)
    print("Got data for %d supernovae" % data["n_sne"])
    def init_wrapped():
        return init_fn(data["n_sne"])

    n_sne = data["n_sne"]
    init_pos = get_truths_labels_significance()
    params = [key[0] for key in init_pos if key[2] is not None]
    params.append("Posterior")
    params.append("weight")
    print("Running single walker, cosmology %d, walk %d" % (n_cosmology, n_run))
    import pystan
    t = stan_dir + "/stan_%d_%d.pkl" % (n_cosmology, n_run)
    sm = pystan.StanModel(file=stan_model, model_name="Cosmology")
    fit = sm.sampling(data=data, iter=n, warmup=w, chains=chains, init=init_wrapped)
    # Dump relevant chains to file
    print("Saving single walker, cosmology %d, walk %d" % (n_cosmology, n_run))
    params = [p for p in params if p in fit.sim["pars_oi"]]
    dictionary = fit.extract(pars=params)
    for key in list(dictionary.keys()):
        print(key, dictionary[key].shape)
        if key.find("log_") == 0:
            dictionary[key[4:]] = np.exp(dictionary[key])
            del dictionary[key]
    if weight_function is not None:
        correction_source = get_correction_data_from_data_source(data_args["data_source"])
        weight_function(dictionary, n_sne, correction_source)
    with open(t, 'wb') as output:
        pickle.dump(dictionary, output)


def run_multiple(data_args, stan_model, stan_dir, n_cosmology, weight_function=None):
    print("Running short steps")
    run_single(data_args, stan_model, stan_dir, n_cosmology, 0, chains=4, weight_function=weight_function, short=True)


def run_cluster(file, stan_output_dir, n_cosmo=15, n_walks=30, n_jobs=30):
    print("Running %s for %d cosmologies, %d walks per cosmology, using %d cores"
          % (file, n_cosmo, n_walks, n_jobs))

    index = n_cosmo * n_walks
    dir_name = os.path.dirname(file)
    h = socket.gethostname()
    from dessn.utility.doJob import write_jobscript, write_jobscript_slurm

    if os.path.exists(stan_output_dir):
        shutil.rmtree(stan_output_dir)
    os.makedirs(stan_output_dir)

    if "smp-cluster" in h:
        filename = write_jobscript(file, name=os.path.basename(dir_name),
                                   num_tasks=index, num_walks=n_walks, num_cpu=n_jobs,
                                   outdir="log", delete=True)
        os.system("qsub %s" % filename)
        print("Submitted SGE job")
    elif "edison" in h or "smp-login" in h:
        partition = "regular" if "edison" in h else "smp"
        filename = write_jobscript_slurm(file, name=os.path.basename(dir_name),
                                         num_tasks=index, num_walks=n_walks, num_cpu=n_jobs,
                                         delete=True, partition=partition)
        os.system("sbatch %s" % filename)
        print("Submitted SLURM job")
    else:
        print("Hostname not recognised as a cluster computer")


def run(data_args, stan_model, filename, weight_function=None):
    h = socket.gethostname()
    assert "data_source" in data_args.keys(), "You must specify a data_source in the data_args!"
    stan_dir = os.path.dirname(os.path.abspath(filename)) + "/stan_output_%s" % data_args["data_source"]
    if not os.path.exists(stan_dir):
        os.makedirs(stan_dir)
    if "science" in h:
        n_cosmology = 0 if len(sys.argv) == 1 else int(sys.argv[1])
        run_multiple(data_args, stan_model, stan_dir, n_cosmology, weight_function=weight_function)
    else:
        if len(sys.argv) == 3:
            i = int(sys.argv[1])
            num_walks_per_cosmology = int(sys.argv[2])
            run_single_input(data_args, stan_model, stan_dir, i,
                             num_walks_per_cosmology=num_walks_per_cosmology,
                             weight_function=weight_function)
        else:
            if len(sys.argv) == 4:
                kwargs = {
                    "n_cosmo": int(sys.argv[1]),
                    "n_walks": int(sys.argv[2]),
                    "n_jobs": int(sys.argv[3])
                }
            else:
                kwargs = {}
            run_cluster(filename, stan_dir, **kwargs)

if __name__ == "__main__":
    print("You probably want to go into a sub directory")
    print("Youll want to give run.py three params: n_cosmo, n_walks, n_jobs")
