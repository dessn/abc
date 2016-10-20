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

    cs = {}
    fs = sorted([f for f in os.listdir(folder) if f.startswith("ston") and f.endswith(".pkl")])
    for f in fs:
        splits = f.split("_")
        c = splits[1]
        t = os.path.abspath(folder + os.sep + f)
        if cs.get(c) is None:
            cs[c] = []
        cs[c].append(get_chain(t, name_map, replace=replace))
    assert len(cs.keys()) > 0, "No results found"
    result = []
    for k in sorted(list(cs.keys())):
        chains = cs[k]
        chain = chains[0]
        for c in chains[1:]:
            for key in chain.keys():
                chain[key] = np.concatenate((chain[key], c[key]))
        posterior = chain["Posterior"]
        bias = chain["sumBias"]
        weights = chain["weights"]
        del chain["weights"]
        del chain["Posterior"]
        del chain["sumBias"]
        result.append((chain, posterior, truths, params, full_params, len(fs), bias, weights))
    return result


if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)

    i = 0
    td = dir_name + "/../output/"
    std = dir_name + "/stan_output"
    results = load_stan_from_folder(std)
    c = ChainConsumer()

    c = ChainConsumer()
    c_all = c.all_colours
    ls = []
    cs = []
    for i, (chain, posterior, truths, params, full_params, num_walks, bias, w) in enumerate(results):
        if i == 5: continue
        print(w.mean(), w.max(), w.min())
        c.add_chain(chain, posterior=posterior, walkers=num_walks, name="Uncorrected %d" % i)
        c.add_chain(chain, posterior=posterior, walkers=num_walks, weights=w, name="Corrected %d" % i)
        ls += ["-", ":"]
        cs += [c_all[i], c_all[i]]
    c.configure_general(linestyles=ls, colours=cs)
    c.plot(filename=td+"approx_plot_full_cosmo.png", truth=truths, parameters=full_params)
