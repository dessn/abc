import os
import pickle
from astropy.cosmology import FlatwCDM
import numpy as np
from numpy.random import normal, uniform
from scipy.stats import skewnorm


def get_truths_labels_significance():
    # Name, Truth, Label, is_significant, min, max
    result = [
        ("Om", 0.3, r"$\Omega_m$", True, 0.1, 0.6),
        ("w", -1.0, r"$w$", True, -1.5, -0.5),
        ("MB", -19.3, r"$M_B$", True, -20, -18.5),
        ("sigma_int", 0.1, r"$\sigma_{\rm int}$", True, 0.05, 0.4),
        ("alpha", 0.1, r"$\alpha$", True, -0.3, 0.5),
        ("beta", 3.0, r"$\beta$", True, 0, 5),
        ("c_loc", 0.1, r"$\langle c \rangle$", False, -0.2, 0.2),
        ("c_scale", 0.1, r"$\sigma_c$", False, 0.05, 0.2),
        ("c_alpha", 2.0, r"$\alpha_c$", False, -2, 2.0),
        ("x1_loc", 0.0, r"$\langle x_1 \rangle$", False, -1.0, 1.0),
        ("x1_scale", 1.0, r"$\sigma_{x1}$", False, 0.1, 2.0),
        ("dscale", 0.08, r"$\delta(0)$", False, -0.2, 0.2),
        ("dratio", 0.5, r"$\delta(\infty)/\delta(0)$", False, 0.0, 1.0),
        ("intrinsic_fractions", np.array([1.0, 0.0, 0.0]), ["$f_{m}$", "$f_{x1}$", "$f_c$"], False,
         None, None),
        ("intrinsic_correlation", np.identity(3), None, False, None, None),
    ]
    return result


def get_physical_data(n_sne=1000, seed=0):
    vals = get_truths_labels_significance()
    mapping = {k[0]: k[1] for k in vals}
    np.random.seed(seed)

    obs_mBx1c = []
    obs_mBx1c_cov = []
    obs_mBx1c_cor = []

    redshifts = np.linspace(0.05, 1.1, n_sne)
    cosmology = FlatwCDM(70.0, mapping["Om"], w0=mapping["w"])
    dist_mod = cosmology.distmod(redshifts).value

    redshift_pre_comp = 0.9 + np.power(10, 0.95 * redshifts)
    MB = mapping["MB"]
    alpha = mapping["alpha"]
    beta = mapping["beta"]
    intrinsic = mapping["sigma_int"]
    dscale = mapping["dscale"]
    dratio = mapping["dratio"]
    p_high_masses = np.random.uniform(low=0.0, high=1.0, size=dist_mod.size)
    fractions = mapping["intrinsic_fractions"][:, None] * mapping["intrinsic_fractions"][None, :]
    correlations = np.dot(mapping["intrinsic_correlation"], mapping["intrinsic_correlation"].T)
    full = fractions * correlations * intrinsic ** 2
    for zz, mu, p in zip(redshift_pre_comp, dist_mod, p_high_masses):

        # Generate the actual mB, x1 and c values
        x1 = normal(loc=mapping["x1_loc"], scale=mapping["x1_scale"])
        c = skewnorm.rvs(mapping["c_alpha"], loc=mapping["c_loc"], scale=mapping["c_scale"])
        if np.random.random() < p:
            mass_correction = dscale * (1.9 * (1 - dratio) / zz + dratio)
        else:
            mass_correction = 0.0
        mb = MB + mu - alpha * x1 + beta * c - mass_correction * p
        vector = np.array([mb, x1, c])
        # Add intrinsic scatter to the mix
        vector += np.random.multivariate_normal([0, 0, 0], full)

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
        "obs_mBx1c_cor": obs_mBx1c_cor,
        "redshifts": redshifts,
        "redshift_pre_comp": redshift_pre_comp,
        "p_high_mass": p_high_masses
    }


def get_analysis_data():
    """ Gets the full analysis data. That is, the observational data, and all the
    useful things we pre-calculate and give to stan to speed things up.
    """
    n_sne = 1000
    data = get_physical_data(n_sne=n_sne, seed=0)
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
    final_redshifts = [z[0] for z in to_sort]
    sorted_vals = [(z[1], i) for i, z in enumerate(to_sort) if z[1] != -1]
    sorted_vals.sort()
    final = [int(z[1] / 2 + 1) for z in sorted_vals]

    update = {
        "n_z": n_z,
        "n_simps": n_simps,
        "zs": final_redshifts,
        "redshift_indexes": final
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
    randoms["true_c"] = cs + normal(scale=0.05, size=n_sne)
    randoms["true_x1"] = cs + normal(scale=0.1, size=n_sne)
    simplex = np.random.random(size=3)
    simplex /= simplex.sum()
    randoms["intrinsic_fractions"] = simplex
    chol = [[1.0, 0.0, 0.0],
            [np.random.random() * 0.1 - 0.05, np.random.random() * 0.1 + 0.7, 0.0],
            [np.random.random() * 0.1 - 0.05, np.random.random() * 0.1 - 0.05,
             np.random.random() * 0.1 + 0.7]]
    randoms["intrinsic_correlation"] = chol
    return randoms


if __name__ == "__main__":
    dir_name = os.path.dirname(__file__) or "."
    output_dir = os.path.abspath(dir_name + "/output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    i = 0
    t = output_dir + "/temp%d.pkl" % i
    data = get_analysis_data()

    # Calculate which parameters we want to keep track of
    init_pos = get_truths_labels_significance()
    params = [key[0] for key in init_pos if key[2] is not None]
    params.append("PointPosteriors")

    # Run that stan
    import pystan

    sm = pystan.StanModel(file="model.stan", model_name="Cosmology")
    fit = sm.sampling(data=data, iter=2500, warmup=1000, chains=4, init=init_fn)

    # Dump relevant chains to file
    with open(t, 'wb') as output:
        dictionary = fit.extract(pars=params)
        pickle.dump(dictionary, output)
