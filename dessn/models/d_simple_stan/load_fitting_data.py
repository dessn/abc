import numpy as np
import inspect
import os

from astropy.cosmology import FlatwCDM
import pandas as pd
import pickle
from dessn.models.d_simple_stan.load_correction_data import load_correction_supernova
from dessn.models.d_simple_stan.truth import get_truths_labels_significance
from scipy.stats import norm


def load_fit_snana_correction(n_sne, include_sim_values=False, directory="snana_passed", zlim=0.3):
    print("Getting SNANA dummy data from %s" % directory)

    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    data_folder = this_dir + "/data/" + directory
    supernovae_files = [np.load(data_folder + "/" + f) for f in os.listdir(data_folder)]
    supernovae = np.vstack(tuple(supernovae_files))

    if zlim is not None:
        redshifts = supernovae[:, 1]
        mask = redshifts < zlim
        supernovae = supernovae[mask, :]

    print("Shuffling data")
    np.random.shuffle(supernovae)

    supernovae = supernovae[:n_sne, :]

    redshifts = supernovae[:, 1]
    apparents = supernovae[:, 6]
    stretches = supernovae[:, 7]
    colours = supernovae[:, 8]

    sapparents = supernovae[:, 3]
    sstretches = supernovae[:, 4]
    scolours = supernovae[:, 5]

    masses = np.ones(supernovae[:, 1].shape)

    print("SHUFFLED: ", apparents.mean(), colours.mean(), stretches.mean())

    obs_mBx1c_cov = []
    obs_mBx1c = []
    actuals = []
    deta_dcalibs = []
    for i, (mb, x1, c, smb, sx1, sc) in enumerate(zip(apparents, stretches, colours, sapparents, sstretches, scolours)):
        vector = np.array([mb, x1, c])
        act = np.array([smb, sx1, sc])
        cov = supernovae[i, 9:9+9].reshape((3, 3))
        # cov = np.diag(np.array([0.05, 0.2, 0.05])**2)
        # vector = act + np.random.multivariate_normal([0, 0, 0], cov)
        calib = supernovae[i, 9+9:].reshape((3, -1))

        obs_mBx1c_cov.append(cov)
        obs_mBx1c.append(vector)
        deta_dcalibs.append(calib)
        actuals.append(act)
    actuals = np.array(actuals)
    covs = np.array(obs_mBx1c_cov)
    deta_dcalibs = np.array(deta_dcalibs)

    result = {
        "n_sne": n_sne,
        "obs_mBx1c": obs_mBx1c,
        "obs_mBx1c_cov": covs,
        "redshifts": redshifts,
        "mass": masses,
        "deta_dcalib": deta_dcalibs
    }
    if include_sim_values:
        result["sim_mBx1c"] = actuals

    return result


def load_fit_snana_diff(n_sne):
    return load_fit_snana_correction(n_sne, directory="diff_passed")


def load_fit_snana_diff2(n_sne):
    return load_fit_snana_correction(n_sne, directory="diff_passed2")


def get_fitres_data():
    print("Getting data from Fitres file")
    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    fitres_file = os.path.abspath(this_dir + "/data/FITOPT000.FITRES")
    dataframe = pd.read_csv(fitres_file, sep='\s+', skiprows=5, comment="#")
    data = dataframe.to_records()

    n_sne = data.size
    obs_mBx1c = data[['mB', 'x1', 'c']].view((float, 3))

    zs = data["zCMB"]
    masses = data["HOST_LOGMASS"]
    gtz = masses > 0
    masses = gtz * masses
    mbs = data["mB"]
    x0s = data["x0"]
    x1s = data["x1"]
    cs = data["c"]

    mbse = data["mBERR"]
    x1se = data["x1ERR"]
    cse = data["cERR"]

    cov_x1_c = data["COV_x1_c"]
    cov_x0_c = data["COV_c_x0"]
    cov_x1_x0 = data["COV_x1_x0"]

    covs = []
    for mb, x0, x1, c, mbe, x1e, ce, cx1c, cx0c, cx1x0 in zip(mbs, x0s, x1s, cs, mbse, x1se, cse, cov_x1_c, cov_x0_c,
                                                              cov_x1_x0):
        cmbx1 = -5 * cx1x0 / (2 * x0 * np.log(10))
        cmbc = -5 * cx0c / (2 * x0 * np.log(10))
        cov = np.array([[mbe * mbe, cmbx1, cmbc], [cmbx1, x1e * x1e, cx1c], [cmbc, cx1c, ce * ce]])
        covs.append(cov)

    return {
        "n_sne": n_sne,
        "obs_mBx1c": obs_mBx1c,
        "obs_mBx1c_cov": covs,
        "redshifts": zs,
        "mass": masses,
        "deta_dcalib": np.zeros((n_sne, 3, 4))
    }


def get_sncosmo_pickle_data(n_sne):
    print("Getting data from sncosmo supernovae pickle")
    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    pickle_file = os.path.abspath(this_dir + "/data/supernovae.pickle")
    with open(pickle_file, 'rb') as pkl:
        supernovae = pickle.load(pkl)
    passed = [s for s in supernovae if s["pc"]]
    np.random.shuffle(passed)
    passed = passed[:n_sne]
    return {
        "n_sne": n_sne,
        "obs_mBx1c": [s["parameters"] for s in passed],
        "obs_mBx1c_cov": [s["covariance"] for s in passed],
        "redshifts": np.array([s["z"] for s in passed]),
        "mass": np.array([s["m"] for s in passed]),
        "deta_dcalib": [s["dp"] for s in passed]
    }


def get_physical_data(n_sne):
    print("Getting simple data")
    vals = get_truths_labels_significance()
    mapping = {k[0]: k[1] for k in vals}

    obs_mBx1c = []
    obs_mBx1c_cov = []
    obs_mBx1c_cor = []
    deta_dcalib = []

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
        deta_dcalib.append(np.ones((3, 4)))

    return {
        "n_sne": n_sne,
        "obs_mBx1c": obs_mBx1c,
        "obs_mBx1c_cov": obs_mBx1c_cov,
        "deta_dcalib": deta_dcalib,
        "redshifts": redshifts,
        "mass": p_high_masses
    }


def get_snana_data():
    print("Getting SNANA data from TEST_DATA_17677")
    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    filename = this_dir + "/data/des_sim.pickle"
    print("Getting SNANA data")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
