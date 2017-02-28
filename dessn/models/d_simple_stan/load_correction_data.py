import numpy as np
import inspect
import os
from scipy.stats import norm


def load_correction_supernova(correction_source, only_passed=True, shuffle=False, zlim=0.3):
    if correction_source == "snana":
        if only_passed:
            result = load_snana_correction(shuffle=shuffle)
        else:
            result = load_snana_failed()
    else:
        raise ValueError("Correction source %s not recognised" % correction_source)
    if zlim is not None:
        mask = result["redshifts"] < zlim
        for key in list(result.keys()):
            result[key] = result[key][mask]
    return result


def load_snana_failed():
    print("Getting SNANA failed data")
    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    data_folder = this_dir + "/data/snana_failed"
    supernovae_files = [np.load(data_folder + "/" + f) for f in os.listdir(data_folder)]
    supernovae = np.vstack(tuple(supernovae_files))
    result = {
        "redshifts": supernovae[:, 0],
        "apparents": supernovae[:, 1]
    }
    return result


def load_snana_correction(shuffle=True):
    print("Getting SNANA correction data")
    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    data_folder = this_dir + "/data/snana_passed"
    supernovae_files = [np.load(data_folder + "/" + f) for f in os.listdir(data_folder)]
    supernovae = np.vstack(tuple(supernovae_files))
    if shuffle:
        print("Shuffling data")
        np.random.shuffle(supernovae)
    result = {
        "masses": np.zeros(supernovae.shape[0]),
        "redshifts": supernovae[:, 1],
        "existing_prob": supernovae[:, 2],
        "apparents": supernovae[:, 3],
        "stretches": supernovae[:, 4],
        "colours": supernovae[:, 5],
    }

    return result