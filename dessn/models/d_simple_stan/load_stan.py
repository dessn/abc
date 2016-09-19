import pickle
import os
from chainconsumer import ChainConsumer
from dessn.models.d_simple_stan.run_stan import get_truths_labels_significance
import itertools
import numpy as np


def get_chain(filename, name_map):
    print("Loading chain from %s" % filename)
    with open(filename, 'rb') as output:
        chain = pickle.load(output)
        keys = list(chain.keys())
        # posterior = chain["PointPosteriors"]
        # del chain["PointPosteriors"]
        for key in keys:
            if key in name_map:
                label = name_map[key]
                if isinstance(label, list):
                    for i, l in enumerate(label):
                        chain[l] = chain[key][:, i]
                else:
                    chain[label] = chain[key]
                del chain[key]
            else:
                new_key = key.replace("_", r"\_")
                if new_key != key:
                    chain[new_key] = chain[key]
                    del chain[key]
    return chain

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)

    i = 0
    td = dir_name + "/output/"
    vals = get_truths_labels_significance()
    full_params = [[k[2]] if not isinstance(k[2], list) else k[2] for k in vals if k[2] is not None]
    params = [[k[2]] if not isinstance(k[2], list) else k[2] for k in vals if k[3] and k[2] is not None]
    full_params = list(itertools.chain.from_iterable(full_params))
    params = list(itertools.chain.from_iterable(params))
    name_map = {k[0]: k[2] for k in vals}
    truths = {k[2]: k[1] for k in vals if not isinstance(k[2], list)}

    fs = [f for f in os.listdir(td) if f.startswith("stan") and f.endswith(".pkl")]
    chains = []
    for f in fs:
        t = os.path.abspath(td + f)
        chains.append(get_chain(t, name_map))
    assert len(chains) > 0, "No results found"
    chain = chains[0]
    for c in chains[1:]:
        for key in chain.keys():
            chain[key] = np.concatenate((chain[key], c[key]))
    # posterior = chain["Posterior"]
    # del chain["Posterior"]
    del chain["PointPosteriors"]
    # full_params = [k[2] for k in vals if k[2] is not None and isinstance(k[1], float)]
    # params = [k[2] for k in vals if k[3] and k[2] is not None and isinstance(k[1], float)]
    # c = ChainConsumer().add_chain(chain, posterior=posterior, walkers=len(fs))
    c = ChainConsumer().add_chain(chain, walkers=len(fs))
    print("Plotting walks")
    c.plot_walks(filename=td+"walk.png")
    print("Plotting surfaces")
    c.plot(filename=td+"plot.png", truth=truths, parameters=params)
    # c.plot(filename=td+"plot_full.png", truth=truths, parameters=full_params)
