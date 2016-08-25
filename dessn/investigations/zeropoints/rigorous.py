from dessn.utility.generator import generate_ia_light_curve, get_summary_stats

import os
import numpy as np
from chainconsumer import ChainConsumer


def get_fit(mabs=-19.3, x1=0.5, c=0.1, z=0.3):
    shallow = np.array([32.46, 32.28, 32.55, 33.12])
    covariance = 0.01 * np.identity(4)
    zps = np.random.multivariate_normal(shallow, covariance)
    lc = generate_ia_light_curve(z, mabs, x1, c, zp=zps, seed=0)
    param, cov = get_summary_stats(z, lc, method="minuit", convert_x0_to_mb=False)
    return param, cov


def get_mean_cov(n_samp=2000):
    ps = []
    cs = []
    for i in range(n_samp):
        p, c = get_fit()
        ps.append(p)
        cs.append(c)
    samples = np.vstack((np.random.multivariate_normal(p, c, size=10000) for p, c in zip(ps, cs)))
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples.T)
    return mean, cov, ps, cs, samples

if __name__ == "__main__":
    temp_dir = os.path.dirname(__file__) + "/output"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    mean, cov, ps, cs, samples = get_mean_cov()
    np.save(temp_dir + "/rigorous.npy", samples)
    c = ChainConsumer()
    c.add_chain(samples, parameters=["$x_0$", "$x_1$", "$c$"])
    c.plot(filename=temp_dir + "/rigorous.png")

