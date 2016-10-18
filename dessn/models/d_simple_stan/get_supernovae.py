import os
import pickle

import numpy as np
from astropy.cosmology import FlatwCDM
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal

from dessn.models.d_simple_stan.simple.run_stan import get_truths_labels_significance
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
            result = get_ia_summary_stats(z, MB_adj, x1, c, do_fit=data, cosmo=cosmology)
            if result is None:
                parameters, cov = None, None
            else:
                parameters, cov = result
            d = {
                "MB": MB,
                "mB": mb,
                "x1": x1,
                "c": c,
                "m": p,
                "z": z,
                "pc": 1 if result is not None else 0,
                "lp": multivariate_normal.logpdf([MB, x1, c], means, pop_cov)
            }
            if data:
                d["covariance"] = cov
                d["parameters"] = parameters
            results.append(d)
            # print("%s nova: %0.2f %0.2f %0.2f %0.3f" % ("PASSED" if result is not None else "failed", MB, x1, c, z))
        except RuntimeError:
            print("Error on nova: %0.2f %0.2f %0.2f %0.3f" % (MB, x1, c, z))
    return results

if __name__ == "__main__":
    n1 = 6000  # samples from which we can draw data
    n2 = 1000000  # samples for Monte Carlo integration of the weights
    jobs = 4  # Using 4 cores
    npr1 = n1 // jobs
    npr2 = n2 // jobs

    dir_name = os.path.dirname(__file__) or "."

    # results1 = Parallel(n_jobs=jobs, max_nbytes="20M", verbose=100)(delayed(get_supernovae)(npr1, True) for i in range(jobs))
    # results1 = [s for r in results1 for s in r]
    # filename1 = os.path.abspath(dir_name + "/output/supernovae.pickle")
    # with open(filename1, 'wb') as output:
    #     pickle.dump(results1, output)
    # print("%d supernova generated for data" % len(results1))

    results2 = Parallel(n_jobs=jobs, max_nbytes="20M", verbose=100)(delayed(get_supernovae)(npr2, False) for i in range(jobs))
    results2 = [s for r in results2 for s in r]
    print("%d supernova generated for weights" % len(results2))
    # filename2 = os.path.abspath(dir_name + "/output/supernovae2.pickle")
    filename3 = os.path.abspath(dir_name + "/output/supernovae2.npy")
    # with open(filename2, 'wb') as output:
    #     pickle.dump(results2, output)
    arr = np.array([[s['MB'], s['mB'], s['x1'], s['c'], s['m'], s['z'], s['pc'], s['lp']] for s in results2]).astype(np.float32)
    print(arr.shape)
    np.save(filename3, arr)
    print("Pickle dumped")
