import numpy as np
import inspect
import os

from astropy.cosmology import FlatwCDM
from scipy.stats import norm, skewnorm, multivariate_normal

from dessn.models.d_simple_stan.truth import get_truths_labels_significance


def load_correction_supernova(correction_source, only_passed=True, shuffle=False, zlim=None):
    if correction_source == "snana":
        if only_passed:
            result = load_snana_correction(shuffle=shuffle)
        else:
            result = load_snana_failed()
    elif correction_source == "simple":
        if only_passed:
            result = get_physical_data(n_sne=10000)
        else:
            result = get_all_physical_data(n_sne=50000)
    else:
        raise ValueError("Correction source %s not recognised" % correction_source)
    if zlim is not None:
        mask = result["redshifts"] < zlim
        for key in list(result.keys()):
            result[key] = result[key][mask]
    return result


def get_physical_data_selection_efficiency(mbs):
    """ Takes an array of mBs, returns list of true or false for if it makes through cut"""
    return np.ones(mbs.shape) == 1
    vals = np.random.uniform(size=mbs.size)

    pdfs = skewnorm.pdf(mbs, -10, 22.5, 5)
    pdfs /= pdfs.max()
    mask = vals < pdfs
    print("%d objects out of %d passed, %d percent" % (mask.sum(), (~mask).sum(), 100*(mask.sum() / mask.size)))

    # print(pdfs.mean(), vals.mean())
    # import matplotlib.pyplot as plt
    # plt.hist(mbs, 50, normed=True)
    # plt.hist(mbs[mask], 50, normed=True)
    # plt.show()
    # exit()

    return mask


def get_physical_data(n_sne):
    data = get_all_physical_data(7 * n_sne)
    mbs = np.array(data["apparents"])
    mask = get_physical_data_selection_efficiency(mbs)
    for key in list(data.keys()):
        if isinstance(data[key], list):
            data[key] = np.array(data[key])[mask][:n_sne]
        elif isinstance(data[key], np.ndarray):
            data[key] = data[key][mask][:n_sne]
    data['n_sne'] = n_sne
    print("Simple data ", data['obs_mBx1c'].shape)
    return data


def get_all_physical_data(n_sne):
    print("Getting all simple data")
    vals = get_truths_labels_significance()
    mapping = {k[0]: k[1] for k in vals}

    obs_mBx1c = []
    obs_mBx1c_cov = []
    obs_mBx1c_cor = []
    deta_dcalib = []

    redshifts = (np.random.uniform(0, 1, n_sne)**0.5)
    cosmology = FlatwCDM(70.0, mapping["Om"]) #, w0=mapping["w"])
    dist_mod = cosmology.distmod(redshifts).value

    redshift_pre_comp = 0.9 + np.power(10, 0.95 * redshifts)
    alpha = mapping["alpha"]
    beta = mapping["beta"]
    dscale = mapping["dscale"]
    dratio = mapping["dratio"]
    # p_high_masses = np.random.uniform(low=-1.0, high=1.0, size=dist_mod.size)
    p_high_masses = np.zeros(shape=dist_mod.shape)
    means = np.array([mapping["mean_MB"], mapping["mean_x1"], mapping["mean_c"]])
    sigmas = np.array([mapping["sigma_MB"], mapping["sigma_x1"], mapping["sigma_c"]])
    sigmas_mat = np.dot(sigmas[:, None], sigmas[None, :])
    correlations = np.dot(mapping["intrinsic_correlation"], mapping["intrinsic_correlation"].T)
    pop_cov = correlations * sigmas_mat
    probs = []
    for zz, mu, p in zip(redshift_pre_comp, dist_mod, p_high_masses):

        # Generate the actual mB, x1 and c values
        MB, x1, c = np.random.multivariate_normal(means, pop_cov)
        probs.append(multivariate_normal.logpdf([MB, x1, c], mean=means, cov=pop_cov))
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
        deta_dcalib.append(np.random.normal(0, 3e-3, size=(3, 8)))

    return {
        "n_sne": n_sne,
        "obs_mBx1c": obs_mBx1c,
        "obs_mBx1c_cov": obs_mBx1c_cov,
        "deta_dcalib": deta_dcalib,
        "redshifts": np.array(redshifts),
        "masses": p_high_masses,
        "existing_prob": probs,
        "apparents": [o[0] for o in obs_mBx1c],
        "stretches": [o[1] for o in obs_mBx1c],
        "colours": [o[2] for o in obs_mBx1c]
    }


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