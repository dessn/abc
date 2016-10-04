import pickle
import os
import numpy as np
from scipy.interpolate import interp1d

from dessn.models.d_simple_stan.run_stan import get_truths_labels_significance
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
        self.sampler = interp1d(cdf, zs)


if __name__ == "__main__":
    redshifts = RedshiftSampler()

    n = 10000  # 10k samples seems a good starting point

    # Redshift distribution
    zs = redshifts.sample(size=n)

    # Population stats
    vals = get_truths_labels_significance()
    mapping = {k[0]: k[1] for k in vals}

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
    for z, p in zip(zs, p_high_masses):
        try:
            MB, x1, c = np.random.multivariate_normal(means, pop_cov)
            mass_correction = dscale * (1.9 * (1 - dratio) / z + dratio)
            MB_adj = MB - alpha * x1 + beta * c - mass_correction * p
            result = get_ia_summary_stats(z, MB_adj, x1, c)
            if result is None:
                parameters, cov = None, None
            else:
                parameters, cov = result
            results.append({
                "MB": MB,
                "x1": x1,
                "c": c,
                "mass": p,
                "redshift": z,
                "passed_cut": result is not None,
                "parameters": parameters,
                "covariance": cov
            })
            print("Generated %s nova: %0.2f %0.2f %0.2f %0.3f" %
                  ("passed" if result is not None else "failed", MB, x1, c, z))
        except RuntimeError:
            print("Error on nova: %0.2f %0.2f %0.2f %0.3f" %
                  (MB, x1, c, z))

    dir_name = os.path.dirname(__file__) or "."
    filename = os.path.abspath(dir_name + "/output/supernovae.pickle")
    with open(filename, 'wb') as output:
        pickle.dump(results, output)
    print("Pickle dumped")