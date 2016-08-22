import pickle
import os
from chainconsumer import ChainConsumer
from dessn.models.d_simple_stan.run_stan import get_truths_labels_significance

dir_name = os.path.dirname(__file__)

i = 0
td = dir_name + "/output/"
t = os.path.abspath(td + "temp%d.pkl" % i)
vals = get_truths_labels_significance()
full_params = [k[2] for k in vals]
params = [k[2] for k in vals if k[3]]

name_map = {k[0]: k[2] for k in vals}
truths = {k[2]: k[1] for k in vals}
with open(t, 'rb') as output:
    chain = pickle.load(output)
    keys = list(chain.keys())
    posterior = chain["PointPosteriors"]
    del chain["PointPosteriors"]
    for key in keys:
        if key in name_map:
            chain[name_map[key]] = chain[key]
            del chain[key]
        else:
            new_key = key.replace("_", r"\_")
            if new_key != key:
                chain[new_key] = chain[key]
                del chain[key]
c = ChainConsumer().add_chain(chain, posterior=posterior)
c.plot(filename=td+"plot_full.png", truth=truths, parameters=full_params)
c.plot(filename=td+"plot.png", truth=truths, parameters=params)
# c.plot_walks(filename=td+"walk.png")