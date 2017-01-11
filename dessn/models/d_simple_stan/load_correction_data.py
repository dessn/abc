import numpy as np
import inspect
import os
from scipy.stats import norm


def load_correction_supernova(correction_source, only_passed=True):
    if correction_source == "snana":
        result = load_snana_correction()
    elif correction_source == "sncosmo":
        result = load_sncosmo_correction()
    else:
        raise ValueError("Correction source %s not recognised" % correction_source)

    if only_passed:
        mask = result["passed"]
        for key in result.keys():
            result[key] = result[key][mask]
    return result


def load_snana_correction():
    print("Getting SNANA correction data")
    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    data_file = this_dir + "/data/correction_snana.npy"
    supernovae = np.load(data_file)
    result = {
        "passed": supernovae[:, 6] > 0.0,
        "masses": np.ones(supernovae.shape[0]),
        "redshifts": supernovae[:, 0],
        "apparents": supernovae[:, 1],
        "stretches": supernovae[:, 2],
        "colours": supernovae[:, 3],
        "smear": supernovae[:, 4]
    }

    # Correct for the way snana does intrinsic dispersion
    result["apparents"] += result["smear"]
    result["existing_prob"] = norm.logpdf(result["colours"], 0, 0.1) \
                              + norm.logpdf(result["stretches"], 0, 1) \
                              + norm.logpdf(result["smear"], 0, 0.1)

    return result


def load_sncosmo_correction():
    print("Getting sncosmo correction data")
    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    pickle_file = this_dir + "/data/supernovae_all.npy"
    supernovae = np.load(pickle_file)
    result = {
        "passed": supernovae[:, 6] == 1,
        "masses": supernovae[:, 4],
        "redshifts": supernovae[:, 5],
        "apparents": supernovae[:, 1],
        "stretches": supernovae[:, 2],
        "colours": supernovae[:, 3],
        "smear": supernovae[:, 4],
        "existing_prob": supernovae[:, 7]
    }
    return result
