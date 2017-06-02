import numpy as np
import emcee
from scipy.stats import norm
from scipy.special import erfc
from joblib import Parallel, delayed


mux, sigmax, alpha = 100, 10, 92
muy, sigmay, beta = 30, 5, 0.2
ndim, nwalkers = 3, 10
epsilon = 4
num = 100

s2 = np.sqrt(2)

def lnprob(theta, xs, ys):
    mux, sigmax, muy = theta
    if sigmax < 0:
        return -np.inf
    return np.sum(norm.logpdf(xs, mux, sigmax) + norm.logpdf(ys, muy, sigmay))

def lnprob2(theta, xs, ys):
    mux, sigmax, muy = theta
    if sigmax < 0:
        return -np.inf
    if mux < alpha:
        return -np.inf
    return np.sum(norm.logpdf(xs, mux, sigmax) + norm.logpdf(ys, muy, sigmay) - np.log(0.5 * erfc((alpha - mux - epsilon)/(s2 * sigmax))))

all_samples, all_sampels_corrected = [], []

def get_data(mux, sigmax, muy, n=1000):
    x = np.random.normal(loc=mux, scale=sigmax, size=n)
    y = np.random.normal(loc=muy, scale=sigmay, size=n)
    mask = (x + beta * y) > alpha
    return x, y, mask
    
def reweight(mux, sigmax, muy):
    original_weight = 0.5 * erfc((alpha - mux - epsilon)/(s2 * sigmax))
    x, y, mask = get_data(mux, sigmax, muy, n=100000)
    new_weight = mask.sum() * 1.0 / mask.size
    diff = np.log(original_weight) - np.log(new_weight)    
    return num * diff    
    #return  -num * np.log(new_weight)    

#val1 = 0.5 * erfc((alpha - mu - epsilon)/(s2 * sigma))
x, y, mask = get_data(mux, sigmax, muy, n=100000)
#val2 = mask.sum() * 1.0 / mask.size
#print(val1, val2)
import matplotlib.pyplot as plt
plt.hist(x, 50, histtype='step', label="all")
plt.hist(x[mask], 50, histtype='step', label="mask1")
plt.legend()
plt.show()

def get_stuff(i):
    np.random.seed(i)
    x, y, mask = get_data(mux, sigmax, muy)
    x = x[mask][:num]
    y = y[mask][:num]

    p0 = [[np.random.normal(100,10), np.random.normal(15, 2), np.random.normal(30, 5)] for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[x,y])
    sampler.run_mcmc(p0, 2000)
    sampler2 = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=[x,y])
    sampler2.run_mcmc(p0, 2000)
    chain1 = sampler.chain[:, 100:, :].reshape((-1, ndim))
    chain2 = sampler2.chain[:, 100:, :].reshape((-1, ndim))
    weights = np.array([reweight(*row) for row in chain2])
    weights -= weights.max()
    weights = np.exp(weights)
    return (chain1, chain2, weights)

res = Parallel(n_jobs=4, backend="threading")(delayed(get_stuff)(i) for i in range(100))
all_samples = np.vstack([r[0] for r in res])
all_sampels_corrected = np.vstack([r[1] for r in res])
weights = np.array([r[2] for r in res]).flatten()
print(weights.max(), weights.min(), weights.mean())

from chainconsumer import ChainConsumer
c = ChainConsumer()
c.add_chain(all_samples, parameters=[r"$\mu$", r"$\sigma$", r"$\mu_y$"], name="Biased")
c.add_chain(all_sampels_corrected, name="Approximate")
c.add_chain(all_sampels_corrected, weights=weights, name="Corrected")
c.configure(flip=False, sigmas=[0,1,2], colors=["#D32F2F", "#4CAF50", "#222222"], linestyles=[":", "--", "-"], shade_alpha=0.2, shade=True)
c.plot(filename="real2.pdf", figsize="column", truth=[100, 10], extents=[[90, 105],[6,15]], parameters=2)