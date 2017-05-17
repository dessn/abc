import numpy as np
import emcee
from scipy.stats import norm
from scipy.special import erfc

mu, sigma, alpha = 100, 10, 82
ndim, nwalkers = 2, 10
num = 50

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

def get_data(mu, sigma, n=1000):
    x = np.random.normal(loc=mu, scale=sigma, size=n)
    mask1 = x > alpha
    mask2 = x > 85 #(mu - 2 * sigma)
    mask = mask1 & mask2
    return x, mask, mask1, mask2
    
def reweight(mu, sigma):
    original_weight = 0.5 * erfc((alpha - mu)/(s2 * sigma))
    x, mask, mask1, mask2 = get_data(mu, sigma, n=100000)
    new_weight = mask.sum() * 1.0 / mask.size
    diff = np.log(original_weight) - np.log(new_weight)    
    return num * diff    

val1 = 0.5 * erfc((alpha - mu)/(s2 * sigma))
x, mask, mask1, mask2 = get_data(mu, sigma, n=100000)
val2 = mask.sum() * 1.0 / mask.size
print(val1, val2)
import matplotlib.pyplot as plt
plt.hist(x, 50, histtype='step', label="all")
plt.hist(x[mask1], 50, histtype='step', label="mask1")
plt.hist(x[mask2], 50, histtype='step', label="mask2")
plt.hist(x[mask], 50, histtype='step', label="both")
plt.legend()
plt.show()
        
    
    
for i in range(100):
    np.random.seed(i)
    
    x, mask, mask1, mask2 = get_data(mu, sigma)
    x = x[mask][:100]
    p0 = [[np.random.normal(100,10), np.random.normal(15, 2)] for i in range(nwalkers)]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[x])
    sampler.run_mcmc(p0, 1000)
    
    sampler2 = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=[x])
    sampler2.run_mcmc(p0, 1000)
    
    all_samples.append(sampler.chain[:, 100:, :].reshape((-1, ndim)))
    all_sampels_corrected.append(sampler2.chain[:, 100:, :].reshape((-1, ndim)))


all_samples = np.vstack(all_samples)
all_sampels_corrected = np.vstack(all_sampels_corrected)
weights = np.array([reweight(*row) for row in all_sampels_corrected])
weights -= weights.max()
weights = np.exp(weights)
print(weights[:20], weights.max(), weights.min(), weights.mean())

from chainconsumer import ChainConsumer
c = ChainConsumer()
c.add_chain(all_samples, parameters=[r"$\mu$", r"$\sigma$"], name="Biased")
c.add_chain(all_sampels_corrected, parameters=[r"$\mu$", r"$\sigma$"], name="Approximate")
c.add_chain(all_sampels_corrected, weights=weights, parameters=[r"$\mu$", r"$\sigma$"], name="Corrected")
c.configure(flip=False, sigmas=[0,1,2], colors=["#D32F2F", "#4CAF50", "#333333"], linestyles=["-", "--", ":"], shade_alpha=0.2, shade=True)
c.plot(filename="real.pdf", figsize="column", truth=[100,10])