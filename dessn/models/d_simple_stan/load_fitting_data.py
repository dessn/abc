import numpy as np
import inspect
import os

from astropy.cosmology import FlatwCDM
import pandas as pd
import pickle
from dessn.models.d_simple_stan.load_correction_data import load_correction_supernova, get_all_physical_data, \
    get_physical_data
from dessn.models.d_simple_stan.truth import get_truths_labels_significance
from scipy.stats import norm, skewnorm


def load_fit_snana_correction(n_sne, include_sim_values=False, directory="snana_passed", zlim=None, shuffle=True):
    print("Getting SNANA dummy data from %s" % directory)

    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    data_folder = this_dir + "/data/" + directory
    supernovae_files = [np.load(data_folder + "/" + f) for f in os.listdir(data_folder)]
    supernovae = np.vstack(tuple(supernovae_files))

    if zlim is not None:
        redshifts = supernovae[:, 1]
        mask = redshifts < zlim
        supernovae = supernovae[mask, :]

    if shuffle:
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

    masses = np.zeros(supernovae[:, 1].shape)

    print("SHUFFLED: ", apparents.mean(), colours.mean(), stretches.mean())

    obs_mBx1c_cov = []
    obs_mBx1c = []
    actuals = []
    deta_dcalibs = []
    for i, (mb, x1, c, smb, sx1, sc) in enumerate(zip(apparents, stretches, colours, sapparents, sstretches, scolours)):
        vector = np.array([mb, x1, c])
        act = np.array([smb, sx1, sc])
        # cov = supernovae[i, 9:9+9].reshape((3, 3))
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
        "masses": masses,
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
        "masses": masses,
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
        "masses": np.array([s["m"] for s in passed]),
        "deta_dcalib": [s["dp"] for s in passed]
    }


def get_fit_physical_data(n_sne):
    return get_physical_data(n_sne)


def get_snana_data():
    print("Getting SNANA data from TEST_DATA_17677")
    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    filename = this_dir + "/data/des_sim.pickle"
    print("Getting SNANA data")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
