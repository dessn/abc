import itertools
import os
import pickle

import numpy as np
from chainconsumer import ChainConsumer

from dessn.models.d_simple_stan.simple.run_stan import get_truths_labels_significance


def get_chain(filename, name_map, replace=True):
    print("Loading chain from %s" % filename)
    with open(filename, 'rb') as output:
        chain = pickle.load(output)
        if replace:
            del chain["intrinsic_correlation"]
            keys = list(chain.keys())
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


def load_stan_from_folder(folder, replace=True):
    vals = get_truths_labels_significance()
    full_params = [[k[2]] if not isinstance(k[2], list) else k[2] for k in vals if k[2] is not None]
    params = [[k[2]] if not isinstance(k[2], list) else k[2] for k in vals if
              k[3] and k[2] is not None]
    full_params = list(itertools.chain.from_iterable(full_params))
    full_params.remove("$\\rho$")
    params = list(itertools.chain.from_iterable(params))
    name_map = {k[0]: k[2] for k in vals}
    truths = {k[2]: k[1] for k in vals if not isinstance(k[2], list)}

    fs = [f for f in os.listdir(folder) if f.startswith("stan") and f.endswith(".pkl")]
    chains = []
    for f in fs:
        t = os.path.abspath(folder + os.sep + f)
        chains.append(get_chain(t, name_map, replace=replace))
    assert len(chains) > 0, "No results found"
    chain = chains[0]
    for c in chains[1:]:
        for key in chain.keys():
            chain[key] = np.concatenate((chain[key], c[key]))
    posterior = chain["Posterior"]
    del chain["Posterior"]
    return chain, posterior, truths, params, full_params, len(fs)


if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)

    i = 0
    td = dir_name + "/../output/"
    std = dir_name + "/stan_output"
    chain, posterior, truths, params, full_params, num_walks = load_stan_from_folder(std)
    c = ChainConsumer().add_chain(chain, posterior=posterior, walkers=num_walks)
    print("Plotting walks")
    c.plot_walks(filename=td+"plot_walk.png")
    print("Plotting surfaces")
    # c.plot(filename=td+"plot.png", truth=truths, parameters=params)
    c.plot(filename=td+"plot_full.png", truth=truths, parameters=full_params)
