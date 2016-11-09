import os
import pickle

import numpy as np
from astropy.cosmology import FlatwCDM
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal

from dessn.models.d_simple_stan.run import get_truths_labels_significance
from dessn.utility.generator import get_ia_summary_stats

"""
The purpose of this file is to generate supernova samples, both for
testing cosmology fitting in STAN, but also bias calculations. I want
to do a simple version in python first so I can confirm everything
works as expected, before moving to SNANA.
"""


class RedshiftSampler(object):
    def __init__(self):
        self.sampler = None

    def sample(self, size=1):
        uniforms = np.random.random(size=size)
        if self.sampler is None:
            self.get_sampler()
        return self.sampler(uniforms)

    def get_sampler(self):
        zs = np.linspace(0.01, 1.2, 10000)

        # These are the rates from the SNANA input files.
        # DNDZ:  POWERLAW2  2.60E-5  1.5  0.0 1.0  # R0(1+z)^Beta Zmin-Zmax
        # DNDZ:  POWERLAW2  7.35E-5  0.0  1.0 2.0
        zlo = zs < 1
        pdf = zlo * 2.6e-5 * (1 + zs) ** 1.5 + (1 - zlo) * 7.35e-5 * (1 + zs)

        # Note you can do the power law analytically, but I don't know the final form
        # of the redshift rate function, so will just do it numerically for now
        cdf = pdf.cumsum()
        cdf = cdf / cdf.max()
        cdf[0] = 0
        self.sampler = interp1d(cdf, zs)


def get_supernovae(n, data=True):
    redshifts = RedshiftSampler()

    # Redshift distribution
    zs = redshifts.sample(size=n)

    # Population stats
    vals = get_truths_labels_significance()
    mapping = {k[0]: k[1] for k in vals}
    cosmology = FlatwCDM(70.0, mapping["Om"])
    mus = cosmology.distmod(zs).value

    alpha = mapping["alpha"]
    beta = mapping["beta"]
    dscale = mapping["dscale"]
    dratio = mapping["dratio"]
    p_high_masses = np.random.uniform(low=0.0, high=1.0, size=n)
    means = np.array([mapping["mean_MB"], mapping["mean_x1"], mapping["mean_c"]])
    sigmas = np.array([mapping["sigma_MB"], mapping["sigma_x1"], mapping["sigma_c"]])
    sigmas_mat = np.dot(sigmas[:, None], sigmas[None, :])
    correlations = np.dot(mapping["intrinsic_correlation"], mapping["intrinsic_correlation"].T)
    pop_cov = correlations * sigmas_mat

    results = []
    for z, p, mu in zip(zs, p_high_masses, mus):
        try:
            MB, x1, c = np.random.multivariate_normal(means, pop_cov)
            mass_correction = dscale * (1.9 * (1 - dratio) / (0.9 + np.power(10, 0.95 * z)) + dratio)
            adjustment = - alpha * x1 + beta * c - mass_correction * p
            MB_adj = MB + adjustment
            mb = MB_adj + mu
            result = get_ia_summary_stats(z, MB_adj, x1, c, cosmo=cosmology, data=data)
            d = {
                "MB": MB,
                "mB": mb,
                "x1": x1,
                "c": c,
                "m": p,
                "z": z,
                "pc": result["passed_cut"],
                "lp": multivariate_normal.logpdf([MB, x1, c], means, pop_cov),
                "dp": result["delta_p"].flatten().tolist() if "delta_p" in results else None,
                "parameters": result.get("params"),
                "covariance": result.get("cov"),
                "lc": result.get("lc")
            }
            results.append(d)
        except RuntimeError:
            print("Error on nova: %0.2f %0.2f %0.2f %0.3f" % (MB, x1, c, z))
    return results


def get_array_from_list_dict(list_dic):
    return np.array([[s['MB'], s['mB'], s['x1'], s['c'], s['m'], s['z'], s['pc'], s['lp']] for s in list_dic]).astype(np.float32)


def is_secure(supernova_dic):
    lc = supernova_dic["lc"]
    bands = lc["band"]
    sn = lc["flux"] / lc["fluxerr"]
    max_sn_per_band = np.array([np.max(sn[bands == b]) for b in np.unique(bands)])
    secure_pass = (max_sn_per_band > 5.3).sum() >= 2
    secure_fail = (max_sn_per_band > 4.7).sum() <= 1
    return secure_pass or secure_fail


def get_data_files(n, data=True):
    supernovae = get_supernovae(n, data=data)
    if data:
        return supernovae

    # Determine which supernova are secure, such that any reasonable change in zp will not affect them
    secure_flag = [is_secure(s) for s in supernovae]
    secures = [sn for sn, sec in zip(supernovae, secure_flag) if sec]
    insecures = [sn for sn, sec in zip(supernovae, secure_flag) if not sec]

    # Turn all secure supernova in nparray
    arr = get_array_from_list_dict(secures)

    # For all unsecure supernova, precalc ston

    for s in insecures:
        lc = s["lc"]
        ston = lc["flux"] / lc["fluxerr"]
        bands = lc["band"]
        s["ston"] = ston
        s["bands"] = bands
        del s["lc"]

    # Combine secure and unsecure into tuple pair
    data = (arr, insecures)
    return data

if __name__ == "__main__":
    n1 = 4000  # samples from which we can draw data
    n2 = 10000  # samples for Monte Carlo integration of the weights
    jobs = 4  # Using 4 cores
    npr1 = n1 // jobs
    npr2 = n2 // jobs

    dir_name = os.path.dirname(__file__) or "."

    if False:
        results1 = Parallel(n_jobs=jobs, max_nbytes="20M", verbose=100)(delayed(get_data_files)(npr1, True) for i in range(jobs))
        results1 = [s for r in results1 for s in r]
        filename1 = os.path.abspath(dir_name + "/output/supernovae.pickle")
        with open(filename1, 'wb') as output:
            pickle.dump(results1, output)
        print("%d supernova generated for data" % len(results1))

    if True:
        results2 = Parallel(n_jobs=jobs, max_nbytes="20M", verbose=100)(delayed(get_data_files)(npr2, False) for i in range(jobs))
        results2 = [s for r in results2 for s in r]
        filename2 = os.path.abspath(dir_name + "/output/supernovae2.pickle")
        with open(filename2, 'wb') as output:
            pickle.dump(results2, output)
        print("%d secure supernova generated for data, %d insecure" % (len(results2[0]), len(results2[1])))
