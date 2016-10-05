import os
import pickle

import shutil
from astropy.cosmology import FlatwCDM
import numpy as np
from numpy.random import normal, uniform
import sys
import platform


def get_truths_labels_significance():
    # Name, Truth, Label, is_significant, min, max
    result = [
        ("Om", 0.3, r"$\Omega_m$", True, 0.1, 0.6),
        # ("w", -1.0, r"$w$", True, -1.5, -0.5),
        ("alpha", 0.1, r"$\alpha$", True, -0.3, 0.5),
        ("beta", 3.0, r"$\beta$", True, 0, 5),
        ("mean_MB", -19.3, r"$\langle M_B \rangle$", True, -20, -18.5),
        ("mean_x1", 0.0, r"$\langle x_1 \rangle$", True, -1.0, 1.0),
        ("mean_c", 0.1, r"$\langle c \rangle$", True, -0.2, 0.2),
        ("sigma_MB", 0.1, r"$\sigma_{\rm m_B}$", True, 0.05, 0.4),
        ("sigma_x1", 0.5, r"$\sigma_{x_1}$", True, 0.1, 2.0),
        ("sigma_c", 0.1, r"$\sigma_c$", True, 0.05, 0.2),
        # ("c_alpha", 2.0, r"$\alpha_c$", False, -2, 2.0),
        ("dscale", 0.08, r"$\delta(0)$", False, -0.2, 0.2),
        ("dratio", 0.5, r"$\delta(\infty)/\delta(0)$", False, 0.0, 1.0),
        ("intrinsic_correlation", np.identity(3), r"$\rho$", False, None, None),
    ]
    return result


def get_pickle_data(n_sne):
    pickle_file = "output/supernovae.pickle"
    with open(pickle_file, 'rb') as pkl:
        supernovae = pickle.load(pkl)
    passed = [s for s in supernovae if s["passed_cut"]][:n_sne]
    return {
        "n_sne": n_sne,
        "obs_mBx1c": [s["parameters"] for s in passed],
        "obs_mBx1c_cov": [s["covariance"] for s in passed],
        "redshifts": np.array([s["redshift"] for s in passed]),
        "mass": np.array([s["mass"] for s in passed])
    }


def get_physical_data(n_sne, seed):
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


def get_snana_data(filename="output/des_sim.pickle"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_analysis_data(sim=False, snana=False):
    """ Gets the full analysis data. That is, the observational data, and all the
    useful things we pre-calculate and give to stan to speed things up.
    """
    n = 500
    if sim:
        data = get_pickle_data(n)
    elif snana:
        data = get_snana_data()
    else:
        data = get_physical_data(n, 2)
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
    dz = redshifts.max() / n_z
    zs = sorted(redshifts.tolist())
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
    n_z = n_sne + len(added_zs)
    n_simps = int((n_z + 1) / 2)
    to_sort = [(z, -1) for z in added_zs] + [(z, i) for i, z in enumerate(redshifts)]
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
        "redshift_pre_comp": 0.9 + np.power(10, 0.95 * redshifts)
    }
    # If you want python2: data.update(update), return data
    return {**data, **update}


def init_fn():
    vals = get_truths_labels_significance()
    randoms = {k[0]: uniform(k[4], k[5]) for k in vals}

    data = get_analysis_data()
    x1s = np.array([x[1] for x in data["obs_mBx1c"]])
    cs = np.array([x[2] for x in data["obs_mBx1c"]])
    n_sne = x1s.size
    randoms["true_MB"] = normal(loc=-19.3, scale=0.1, size=n_sne)
    randoms["true_c"] = cs + normal(scale=0.05, size=n_sne)
    randoms["true_x1"] = cs + normal(scale=0.1, size=n_sne)
    chol = [[1.0, 0.0, 0.0],
            [np.random.random() * 0.1 - 0.05, np.random.random() * 0.1 + 0.7, 0.0],
            [np.random.random() * 0.1 - 0.05, np.random.random() * 0.1 - 0.05,
             np.random.random() * 0.1 + 0.7]]
    randoms["intrinsic_correlation"] = chol
    return randoms


if __name__ == "__main__":
    file = os.path.abspath(__file__)
    dir_name = os.path.dirname(__file__) or "."
    stan_output_dir = os.path.abspath(dir_name + "/stan_output")
    output_dir = os.path.abspath(dir_name + "/output")
    t = stan_output_dir + "/stan.pkl"
    data = get_analysis_data(output_dir)

    # Calculate which parameters we want to keep track of
    init_pos = get_truths_labels_significance()
    params = [key[0] for key in init_pos if key[2] is not None]
    params.append("Posterior")
    if len(sys.argv) == 2:
        print("Running single walker")
        # Assuming linux environment for single thread
        dessn_dir = file[: file.index("dessn")]
        sys.path.append(dessn_dir)
        import pystan
        i = int(sys.argv[1])
        t = stan_output_dir + "/stan%d.pkl" % i
        sm = pystan.StanModel(file="model.stan", model_name="Cosmology")
        fit = sm.sampling(data=data, iter=2000, warmup=1000, chains=1, init=init_fn)

        # Dump relevant chains to file
        with open(t, 'wb') as output:
            dictionary = fit.extract(pars=params)
            pickle.dump(dictionary, output)
    else:
        # Run that stan locally
        if "centos" in platform.platform():
            # Assuming this is obelix
            dessn_dir = file[: file.index("dessn")]
            sys.path.append(dessn_dir)
            from dessn.utility.doJob import write_jobscript
            if len(sys.argv) == 3:
                num_walks = int(sys.argv[1])
                num_jobs = int(sys.argv[2])
            else:
                num_walks = 30
                num_jobs = 30
            if os.path.exists(stan_output_dir):
                shutil.rmtree(stan_output_dir)
            os.makedirs(stan_output_dir)
            filename = write_jobscript(file, num_walks=num_walks, num_cpu=num_jobs, delete=True)
            os.system("qsub %s" % filename)
            print("Called qsub")
        else:
            print("Running short steps")
            # Assuming its my laptop vbox
            import pystan
            sm = pystan.StanModel(file="model.stan", model_name="Cosmology")
            fit = sm.sampling(data=data, iter=1300, warmup=800, chains=4, init=init_fn)

            # Dump relevant chains to file
            with open(t, 'wb') as output:
                dictionary = fit.extract(pars=params)
                pickle.dump(dictionary, output)
