import os
import pickle
import inspect

import shutil
import numpy as np
from numpy.random import uniform
import sys
import socket
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
from dessn.models.d_simple_stan.get_cosmologies import get_cosmology_dictionary
from dessn.models.d_simple_stan.load_correction_data import load_correction_supernova
from dessn.models.d_simple_stan.load_fitting_data import get_sncosmo_pickle_data, load_fit_snana_correction, \
    get_fitres_data, get_physical_data, get_snana_data, load_fit_snana_diff
from dessn.models.d_simple_stan.truth import get_truths_labels_significance


def get_simulation_data(correction_source="snana", n=5000):
    print("Getting %s simulation data" % correction_source)
    supernovae = load_correction_supernova(correction_source=correction_source)
    redshifts = supernovae["redshifts"][:n]
    apparents = supernovae["apparents"][:n]
    stretches = supernovae["stretches"][:n]
    colours = supernovae["colours"][:n]
    existing_prob = supernovae["existing_prob"][:n]
    masses = supernovae["masses"][:n]

    obs_mBx1c = []
    for mb, x1, c in zip(apparents, stretches, colours):
        vector = np.array([mb, x1, c])
        obs_mBx1c.append(vector)

    return {
        "n_sim": n,
        "sim_mBx1c": obs_mBx1c,
        "sim_redshifts": redshifts,
        "sim_mass": masses,
        "sim_log_prob": existing_prob
    }


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

        # dscale = chain_dictionary["dscale"][i]
        # dratio = chain_dictionary["dratio"][i]
        # redshift_pre_comp = 0.9 + np.power(10, 0.95 * redshifts)
        # mass_correction = dscale * (1.9 * (1 - dratio) / redshift_pre_comp + dratio)
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
        weight.append(reweight)

    weights = np.array(weight)
    return weights


def add_weight_to_chain(chain_dictionary, n_sne, correction_source, num=None, trim=False, trim_v=-10, shuffle=False):
    # Load supernova for correction
    supernovae = load_correction_supernova(correction_source=correction_source, shuffle=shuffle)
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


def get_gp_data(n_sne, add_gp, seed=0, correction_source="snana", num=None):
    np.random.seed(seed)
    inits = [init_fn(n_sne) for i in range(add_gp)]
    keys = inits[0].keys()
    d = {k: np.array([i[k] for i in inits]) for k in keys}
    d["weight"] = np.ones(add_gp)
    add_weight_to_chain(d, n_sne, trim=False, shuffle=False, correction_source=correction_source, num=num)

    vals = d["calc_weight"]
    vals *= n_sne
    print("Get data has weight mean and std of ", vals.mean(), np.std(vals))

    keys_to_remove = ["Posterior", "weight", "old_weight", "dscale",
                      "dratio", "calibration", "calc_weight", "deviations"]
    keys = [k for k in sorted(d.keys()) if k not in keys_to_remove]
    for key in keys:
        v = d[key]
        if len(v.shape) > 1:
            if len(v.shape) > 2:
                col = []
                for item in v:
                    item = item.dot(item.T)
                    col.append([item[0, 1], item[0, 2], item[1, 2]])
                v = np.array(col)
            else:
                v = v.reshape((v.shape[0], -1))
        else:
            v = np.atleast_2d(v).T
        d[key] = v

    flat = np.hstack(tuple([d[k] for k in keys]))
    return flat, vals, keys


def get_base_data(data_source, n):
    if data_source == "sncosmo":
        return get_sncosmo_pickle_data(n)
    elif data_source == "snana_dummy":
        return load_fit_snana_correction(n)
    elif data_source == "snana_diff":
        return load_fit_snana_diff(n)
    elif data_source == "snana":
        return get_snana_data()
    elif data_source == "fitres":
        return get_fitres_data()
    elif data_source == "simple":
        return get_physical_data(n)
    else:
        raise ValueError("Data source %s not recognised" % data_source)


def get_correction_data_from_data_source(data_source):
    if data_source == "sncosmo":
        return "sncosmo"
    elif data_source == "snana_dummy":
        return "snana"
    elif data_source == "snana":
        return "snana"
    elif data_source == "fitres":
        return "snana"
    elif data_source == "simple":
        return None
    else:
        raise ValueError("Data source %s not recognised" % data_source)


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
    n_z = 1000
    if add_sim:
        correction_source = get_correction_data_from_data_source(data_source)
        sim_data = get_simulation_data(correction_source=correction_source, n=add_sim)
        n_sim = sim_data["n_sim"]
        sim_redshifts = sim_data["sim_redshifts"]

        dz = max(redshifts.max(), sim_redshifts.max()) / n_z
        zs = sorted(redshifts.tolist() + sim_redshifts.tolist())
    else:
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
    if add_sim:
        to_sort += [(z, -1, i) for i, z in enumerate(sim_redshifts)]
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
        "calib_std": np.ones(4) * 0.01
    }

    if add_sim:
        sim_sorted_vals = [(z[2], i) for i, z in enumerate(to_sort) if z[2] != -1]
        sim_sorted_vals.sort()
        sim_final = [int(z[1] / 2 + 1) for z in sim_sorted_vals]
        update["sim_redshift_indexes"] = sim_final
        update["sim_redshift_pre_comp"] = 0.9 + np.power(10, 0.95 * sim_redshifts)

    if extra_args is None:
        extra_args = {}

    # If you want python2: data.update(update), return data
    return {**data, **update, **sim_data, **extra_args}


def init_fn(n_sne):
    vals = get_truths_labels_significance()
    randoms = {k[0]: uniform(k[4], k[5]) for k in vals}
    randoms["deviations"] = np.random.normal(scale=0.2, size=(n_sne, 3))
    chol = [[1.0, 0.0, 0.0],
            [np.random.random() * 0.1 - 0.05, np.random.random() * 0.1 + 0.7, 0.0],
            [np.random.random() * 0.1 - 0.05, np.random.random() * 0.1 - 0.05,
             np.random.random() * 0.1 + 0.7]]
    randoms["intrinsic_correlation"] = chol
    randoms["calibration"] = (np.random.uniform(size=4) - 0.5) * 0.0001
    return randoms


def run_single_input(data_args, stan_model, stan_dir, i, num_walks_per_cosmology=20, weight_function=None):
    n_cosmology = i // num_walks_per_cosmology
    n_run = i % num_walks_per_cosmology
    run_single(data_args, stan_model, stan_dir, n_cosmology, n_run, weight_function=weight_function)


def run_single(data_args, stan_model, stan_dir, n_cosmology, n_run, chains=1, weight_function=None, short=False):
    if short:
        w, n = 500, 1000
    else:
        w, n = 1000, 5000
    data = get_analysis_data(seed=n_cosmology, **data_args)

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
    with open(t, 'wb') as output:
        params = [p for p in params if p in fit.sim["pars_oi"]]
        dictionary = fit.extract(pars=params)
        if weight_function is not None:
            correction_source = get_correction_data_from_data_source(data_args["data_source"])
            weight_function(dictionary, n_sne, correction_source)
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