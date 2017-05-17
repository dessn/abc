import numpy as np
import emcee
from scipy.stats import norm


mu, sigma = 100, 10
ndim, nwalkers = 2, 10

def lnprob(theta, data):
    if theta[1] < 0:
        return -np.inf
    return np.sum(norm.logpdf(data, theta[0], theta[1]))

all_samples = []
for i in range(100):
    np.random.seed(i)
    x = np.random.normal(loc=mu, scale=sigma, size=100)
    p0 = [[np.random.normal(100,10), np.random.normal(15, 2)] for i in range(nwalkers)]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[x])
    sampler.run_mcmc(p0, 2000)
    all_samples.append(sampler.chain[:, 100:, :].reshape((-1, ndim)))


all_samples = np.vstack(all_samples)

from chainconsumer import ChainConsumer
c = ChainConsumer()
c.add_chain(all_samples, parameters=[r"$\mu$", r"$\sigma$"])
c.configure(flip=False, sigmas=[0,1,2])
c.plot(filename="perfect.pdf", figsize="column", truth=[100,10])

