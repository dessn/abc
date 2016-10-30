import os
import pickle
import inspect

import shutil
from astropy.cosmology import FlatwCDM
import numpy as np
from numpy.random import uniform
import sys
import socket


def get_truths_labels_significance():
    # Name, Truth, Label, is_significant, min, max
    result = [
        ("Om", 0.3, r"$\Omega_m$", True, 0.1, 0.6),
        # ("w", -1.0, r"$w$", True, -1.5, -0.5),
        ("alpha", 0.1, r"$\alpha$", True, 0, 0.5),
        ("beta", 3.0, r"$\beta$", True, 0, 5),
        ("mean_MB", -19.3, r"$\langle M_B \rangle$", True, -19.6, -18.8),
        ("mean_x1", 0.0, r"$\langle x_1 \rangle$", True, -1.0, 1.0),
        ("mean_c", 0.1, r"$\langle c \rangle$", True, -0.2, 0.2),
        ("sigma_MB", 0.1, r"$\sigma_{\rm m_B}$", True, 0.05, 0.4),
        ("sigma_x1", 0.5, r"$\sigma_{x_1}$", True, 0.1, 1.0),
        ("sigma_c", 0.1, r"$\sigma_c$", True, 0.05, 0.2),
        # ("c_alpha", 2.0, r"$\alpha_c$", False, -2, 2.0),
        ("dscale", 0.08, r"$\delta(0)$", False, -0.2, 0.2),
        ("dratio", 0.5, r"$\delta(\infty)/\delta(0)$", False, 0.0, 1.0),
        ("intrinsic_correlation", np.identity(3), r"$\rho$", False, None, None),
    ]
    return result


def get_pickle_data(n_sne, seed=0, zt=10.0):
    print("Getting data from supernovae pickle")
    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    pickle_file = os.path.abspath(this_dir + "/output/supernovae.pickle")
    with open(pickle_file, 'rb') as pkl:
        supernovae = pickle.load(pkl)
    passed = [s for s in supernovae if s["pc"] and s["z"] < zt]
    np.random.seed(seed)
    np.random.shuffle(passed)
    passed = passed[:n_sne]
    return {
        "n_sne": n_sne,
        "obs_mBx1c": [s["parameters"] for s in passed],
        "obs_mBx1c_cov": [s["covariance"] for s in passed],
        "redshifts": np.array([s["z"] for s in passed]),
        "mass": np.array([s["m"] for s in passed])
    }


def get_simulation_data(n=5000):
    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    pickle_file = this_dir + "/output/supernovae2.npy"
    supernovae = np.load(pickle_file)
    mask = supernovae[:, 6] == 1
    supernovae = supernovae[mask, :]
    if n == -1:
        n = supernovae.shape[0]
    supernovae = supernovae[:n, :]
    return {
        "n_sim": n,
        "sim_mBx1c": supernovae[:, 1:4],
        "sim_log_prob": supernovae[:, 7],
        "sim_redshifts": supernovae[:, 5],
        "sim_mass": supernovae[:, 4]
    }


def get_physical_data(n_sne, seed=0):
    print("Getting simple data")
    vals = get_truths_labels_significance()
    mapping = {k[0]: k[1] for k in vals}
    np.random.seed(seed)

    obs_mBx1c = []
    obs_mBx1c_cov = []
    obs_mBx1c_cor = []

    redshifts = np.linspace(0.05, 1.1, n_sne)
    cosmology = FlatwCDM(70.0, mapping["Om"]) #, w0=mapping["w"])
    dist_mod = cosmology.distmod(redshifts).value

    redshift_pre_comp = 0.9 + np.power(10, 0.95 * redshifts)
    alpha = mapping["alpha"]
    beta = mapping["beta"]
    dscale = mapping["dscale"]
    dratio = mapping["dratio"]
    p_high_masses = np.random.uniform(low=0.0, high=1.0, size=dist_mod.size)
    means = np.array([mapping["mean_MB"], mapping["mean_x1"], mapping["mean_c"]])
    sigmas = np.array([mapping["sigma_MB"], mapping["sigma_x1"], mapping["sigma_c"]])
    sigmas_mat = np.dot(sigmas[:, None], sigmas[None, :])
    correlations = np.dot(mapping["intrinsic_correlation"], mapping["intrinsic_correlation"].T)
    pop_cov = correlations * sigmas_mat
    for zz, mu, p in zip(redshift_pre_comp, dist_mod, p_high_masses):

        # Generate the actual mB, x1 and c values
        MB, x1, c = np.random.multivariate_normal(means, pop_cov)
        mass_correction = dscale * (1.9 * (1 - dratio) / zz + dratio)
        mb = MB + mu - alpha * x1 + beta * c - mass_correction * p
        vector = np.array([mb, x1, c])
        # Add intrinsic scatter to the mix
        diag = np.array([0.05, 0.3, 0.05]) ** 2
        cov = np.diag(diag)
        vector += np.random.multivariate_normal([0, 0, 0], cov)
        cor = cov / np.sqrt(np.diag(cov))[None, :] / np.sqrt(np.diag(cov))[:, None]
        obs_mBx1c_cor.append(cor)
        obs_mBx1c_cov.append(cov)
        obs_mBx1c.append(vector)

    return {
        "n_sne": n_sne,
        "obs_mBx1c": obs_mBx1c,
        "obs_mBx1c_cov": obs_mBx1c_cov,
        "redshifts": redshifts,
        "mass": p_high_masses
    }


def get_snana_data():
    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    filename = this_dir + "/output/des_sim.pickle"
    print("Getting SNANA data")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_analysis_data(sim=True, snana=False, seed=0, add_sim=0, **extra_args):
    """ Gets the full analysis data. That is, the observational data, and all the
    useful things we pre-calculate and give to stan to speed things up.
    """
    n = 500
    if sim:
        data = get_pickle_data(n, seed=seed)
    elif snana:
        data = get_snana_data()
    else:
        data = get_physical_data(n, seed=seed)
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
        sim_data = get_simulation_data(n=add_sim)
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


def init_fn():
    vals = get_truths_labels_significance()
    randoms = {k[0]: uniform(k[4], k[5]) for k in vals}
    data = get_analysis_data()
    mass = data["mass"]
    randoms["deviations"] = np.random.normal(scale=0.2, size=(mass.size, 3))
    chol = [[1.0, 0.0, 0.0],
            [np.random.random() * 0.1 - 0.05, np.random.random() * 0.1 + 0.7, 0.0],
            [np.random.random() * 0.1 - 0.05, np.random.random() * 0.1 - 0.05,
             np.random.random() * 0.1 + 0.7]]
    randoms["intrinsic_correlation"] = chol
    return randoms


def run_single_input(data_args, stan_model, i, num_walks_per_cosmology=20, weight_function=None):
    n_cosmology = i // num_walks_per_cosmology
    n_run = i % num_walks_per_cosmology
    run_single(data_args, stan_model, n_cosmology, n_run, weight_function=weight_function)


def run_single(data_args, stan_model, n_cosmology, n_run, chains=1, weight_function=None, short=False):
    if short:
        w, n = 1000, 2000
    else:
        w, n = 2000, 10000
    data = get_analysis_data(seed=n_cosmology, **data_args)
    # import matplotlib.pyplot as plt
    # plt.hist([a[0] for a in data["obs_mBx1c"]], 30)
    # plt.axvline(24.33)
    # plt.figure()
    # plt.hist(data["redshifts"], 30)
    # plt.show()
    # exit()
    n_sne = data["n_sne"]
    init_pos = get_truths_labels_significance()
    params = [key[0] for key in init_pos if key[2] is not None]
    params.append("Posterior")
    params.append("weight")
    print("Running single walker, cosmology %d, walk %d" % (n_cosmology, n_run))
    import pystan
    dir_name = os.path.dirname(stan_model)
    t = dir_name + "/stan_output/stan_%d_%d.pkl" % (n_cosmology, n_run)
    sm = pystan.StanModel(file=stan_model, model_name="Cosmology")
    fit = sm.sampling(data=data, iter=n, warmup=w, chains=chains, init=init_fn)
    # Dump relevant chains to file
    print("Saving single walker, cosmology %d, walk %d" % (n_cosmology, n_run))
    with open(t, 'wb') as output:
        dictionary = fit.extract(pars=params)
        if weight_function is not None:
            weight_function(dictionary, n_sne)
        pickle.dump(dictionary, output)


def get_mc_simulation_data():
    pickle_file = os.path.dirname(inspect.stack()[0][1]) + "/output/supernovae2.npy"
    supernovae = np.load(pickle_file)

    return {
        "n_sim": supernovae.shape[0],
        "sim_MB": supernovae[:, 0],
        "sim_mB": supernovae[:, 1],
        "sim_x1": supernovae[:, 2],
        "sim_c": supernovae[:, 3],
        "sim_passed": supernovae[:, 6],
        "sim_log_prob": supernovae[:, 7],
        "sim_redshift": supernovae[:, 5],
        "sim_mass": supernovae[:, 4]
    }


def run_multiple(data_args, stan_model, n_cosmology, weight_function=None):
    print("Running short steps")
    run_single(data_args, stan_model, n_cosmology, 0, chains=4, weight_function=weight_function, short=True)


def run_cluster(file, n_cosmo=15, n_walks=30, n_jobs=30):
    print("Running %s for %d cosmologies, %d walks per cosmology, using %d cores"
          % (file, n_cosmo, n_walks, n_jobs))

    index = n_cosmo * n_walks
    dir_name = os.path.dirname(file)
    stan_output_dir = dir_name + "/stan_output"
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
    elif "edison" in h:
        filename = write_jobscript_slurm(file, name=os.path.basename(dir_name),
                                         num_tasks=index, num_walks=n_walks, num_cpu=n_jobs,
                                         delete=True)
        os.system("sbatch %s" % filename)
        print("Submitted SLURM job")
    else:
        print("Hostname not recognised as a cluster computer")


def run(data_args, stan_model, filename, weight_function=None):
    h = socket.gethostname()
    if "science" in h:
        n_cosmology = 0 if len(sys.argv) == 1 else int(sys.argv[1])
        run_multiple(data_args, stan_model, n_cosmology, weight_function=weight_function)
    else:
        if len(sys.argv) == 3:
            i = int(sys.argv[1])
            num_walks_per_cosmology = int(sys.argv[2])
            run_single_input(data_args, stan_model, i,
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
            run_cluster(filename, **kwargs)

if __name__ == "__main__":
    print("You probably want to go into a sub directory")
    print("Youll want to give run.py three params: n_cosmo, n_walks, n_jobs")