import numpy as np
import emcee
from scipy.stats import norm
from scipy.special import erfc

mu, sigma, alpha = 100, 10, 85
ndim, nwalkers = 2, 10
s2 = np.sqrt(2)
def lnprob(theta, data):
    if theta[1] < 0:
        return -np.inf
    return np.sum(norm.logpdf(data, theta[0], theta[1]))

def lnprob2(theta, data):
    mu, sigma = theta
    if sigma < 0:
        return -np.inf
    if mu < alpha:
        return -np.inf
    return np.sum(norm.logpdf(data, mu, sigma) - np.log(0.5 * erfc((alpha - mu)/(s2 * sigma))))

all_samples, all_sampels_corrected = [], []
for i in range(100):
    np.random.seed(i)
    
    x = np.random.normal(loc=mu, scale=sigma, size=1000)
    mask = x > alpha
    x = x[mask][:100]
    p0 = [[np.random.normal(100,10), np.random.normal(15, 2)] for i in range(nwalkers)]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[x])
    sampler.run_mcmc(p0, 2000)
    
    sampler2 = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=[x])
    sampler2.run_mcmc(p0, 2000)
    
    all_samples.append(sampler.chain[:, 100:, :].reshape((-1, ndim)))
    all_sampels_corrected.append(sampler2.chain[:, 100:, :].reshape((-1, ndim)))


all_samples = np.vstack(all_samples)
all_sampels_corrected = np.vstack(all_sampels_corrected)

from chainconsumer import ChainConsumer
c = ChainConsumer()
c.add_chain(all_samples, parameters=[r"$\mu$", r"$\sigma$"], name="Biased")
c.add_chain(all_sampels_corrected, parameters=[r"$\mu$", r"$\sigma$"], name="Corrected")
c.configure(flip=False, sigmas=[0,1,2], colors=["#D32F2F", "#4CAF50"], linestyles=["-", "--"], shade_alpha=0.2)
c.plot(filename="imperfect.pdf", figsize="column", truth=[100,10], extents=[[95, 105],[7,14]])