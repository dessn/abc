import pickle
import os
from chainconsumer import ChainConsumer

dir_name = os.path.dirname(__file__)

i = 0
td = dir_name + "/output/"
t = os.path.abspath(td + "temp%d.pkl" % i)
name_map = {
    "Om": r"$\Omega_m$",
    "w": "$w$",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "MB": "$M_B$",
    "sigma_int": r"$\sigma_{\rm int}$"
}
with open(t, 'rb') as output:
    chain = pickle.load(output)
    keys = list(chain.keys())
    posterior = chain["PointPosteriors"]
    del chain["PointPosteriors"]
    for key in keys:
        if key in name_map:
            chain[name_map[key]] = chain[key]
            del chain[key]

c = ChainConsumer().add_chain(chain, posterior=posterior)
truths = {
    "$M_B$": -19.3,
    r"$\Omega_m$": 0.3,
    "$w$": -1,
    r"$\sigma_{\rm int}$": 0.1,
    r"$\alpha$": 0.1,
    r"$\beta$": 3
}
c.plot(filename=td+"plot.png", truth=truths)
# c.plot_walks(filename=td+"walk.png")