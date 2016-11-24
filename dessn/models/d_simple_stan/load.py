import itertools
import os
import pickle
import inspect
import numpy as np
from chainconsumer import ChainConsumer

from dessn.models.d_simple_stan.run import get_truths_labels_significance


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


def load_stan_from_folder(folder, replace=True, merge=True, cut=True, num=None):
    vals = get_truths_labels_significance()
    full_params = [[k[2]] if not isinstance(k[2], list) else k[2] for k in vals if k[2] is not None]
    params = [[k[2]] if not isinstance(k[2], list) else k[2] for k in vals if
              k[3] and k[2] is not None]
    full_params = list(itertools.chain.from_iterable(full_params))
    full_params.remove("$\\rho$")
    params = list(itertools.chain.from_iterable(params))
    name_map = {k[0]: k[2] for k in vals}
    truths = {k[2]: k[1] for k in vals if not isinstance(k[2], list)}
    is_array = [k[0] for k in vals if not isinstance(k[1], float) and not isinstance(k[1], int)]
    cs = {}
    fs = sorted([f for f in os.listdir(folder) if f.startswith("stan") and f.endswith(".pkl")])
    if num is not None:
        filter = "_%d_" % num
        fs = [f for f in fs if filter in f]

    for f in fs:
        splits = f.split("_")
        c = splits[1]
        t = os.path.abspath(folder + os.sep + f)
        if cs.get(c) is None:
            cs[c] = []
        cs[c].append(get_chain(t, name_map, replace=replace))
    assert len(cs.keys()) > 0, "No results found"
    result = []
    good_ks = []
    for k in sorted(list(cs.keys())):
        chains = cs[k]
        chain = chains[0]
        for c in chains[1:]:
            for key in chain.keys():
                chain[key] = np.concatenate((chain[key], c[key]))
        posterior = chain["Posterior"]
        del chain["Posterior"]
        if "weight" in chain.keys():
            weights = chain["weight"]
            del chain["weight"]
        else:
            weights = np.ones(posterior.shape)
        if "old\\_weight" in chain.keys():
            ow = chain["old\\_weight"]
            ow -= ow.min()
            ow = np.exp(ow)
            del chain["old\\_weight"]
        else:
            ow = np.ones(posterior.shape)
        print(chain.keys())
        for param in is_array:
            latex = name_map[param]
            truth_val = truths[latex]
            shape = truth_val.shape
            if len(shape) > 1 or latex not in chain: continue  # Dont do 2D parameters
            for i in range(shape[0]):
                column = chain[latex][:, i]
                chain[latex % i] = column
                truths[latex % i] = truth_val[i]
            del chain[latex]

        c = ChainConsumer()
        c.add_chain(chain, weights=weights)
        summary = c.get_summary()
        num_failed = sum([1 if summary[k][0] is None else 0 for k in summary.keys()])
        num_param = len(list(summary.keys()))
        if not cut or num_failed < 4:
            print("Chain %s good" % k)
            good_ks.append(k)
            result.append((chain, posterior, truths, params, full_params, len(chains), weights, ow))
        else:
            print("Chain %s is bad" % k)
    if merge:
        rr = list(result[0])
        for r in result[1:]:
            for key in rr[0].keys():
                rr[0][key] = np.concatenate((rr[0][key], r[0][key]))
            rr[1] = np.concatenate((rr[1], r[1]))
            rr[6] = np.concatenate((rr[6], r[6]))
            rr[7] = np.concatenate((rr[7], r[7]))
            rr[5] += r[5]
        return tuple(rr)
    else:
        return result


def plot_all(folder, output, output_walk=None):
    """ Plot all chains as one """
    print("Plotting all as one")
    chain, posterior, t, p, f, l, w, ow = load_stan_from_folder(folder, merge=True)
    c = ChainConsumer()
    c.add_chain(chain, weights=w, posterior=posterior, walkers=l)
    c.plot(filename=output, truth=t)
    if output_walk is not None:
        c.plot_walks(filename=output_walk)


def plot_single_cosmology(folder, output, i=0, output_walk=None):
    print("Plotting cosmology realisation %d" % i)
    res = load_stan_from_folder(folder, merge=False)
    c = ChainConsumer()
    chain, posterior, t, p, f, l, w, ow = res[i]
    c.add_chain(chain, weights=w, posterior=posterior, walkers=l, name="%d"%i)
    c.plot(filename=output, truth=t)
    if output_walk is not None:
        c.plot_walks(filename=output_walk)


def plot_single_cosmology_weight(folder, output, i=0):
    print("Plotting cosmology realisation %d" % i)
    res = load_stan_from_folder(folder, merge=False)
    c = ChainConsumer()
    chain, posterior, t, p, f, l, w, ow = res[i]
    c.add_chain(chain, posterior=posterior, walkers=l, name="Uncorrected %d"%i)
    c.add_chain(chain, weights=w, posterior=posterior, walkers=l, name="Corrected %d"%i)
    c.plot(filename=output, truth=t)


def plot_all_weight(folder, output):
    """ Plot all chains as one, with and without weights applied """
    print("Plotting all as one, with old and new weights")
    chain, posterior, t, p, f, l, w, ow = load_stan_from_folder(folder, merge=True)
    c = ChainConsumer()
    c.add_chain(chain, posterior=posterior, walkers=l, name="Uncorrected")
    c.add_chain(chain, weights=w, posterior=posterior, walkers=l, name="Corrected")
    c.plot(filename=output, truth=t)


def plot_all_no_weight(folder, output):
    """ Plot all chains as one, with and without weights applied """
    print("Plotting all as one, with old and new weights")
    chain, posterior, t, p, f, l, w, ow = load_stan_from_folder(folder, merge=True)
    c = ChainConsumer()
    c.add_chain(chain, posterior=posterior, walkers=l)
    c.plot(filename=output, truth=t)


def plot_separate(folder, output):
    """ Plot separate cosmologies """
    print("Plotting all cosmologies separately")
    res = load_stan_from_folder(folder, merge=False)
    c = ChainConsumer()
    for i, (chain, posterior, t, p, f, l, w, ow) in enumerate(res):
        c.add_chain(chain, weights=w, posterior=posterior, walkers=l, name="%d"%i)
    c.plot(filename=output, truth=t)


def plot_separate_weight(folder, output):
    """ Plot separate cosmologies, with and without weights applied """
    print("Plotting all cosmologies separately, with old and new weights")
    res = load_stan_from_folder(folder, merge=False)
    c = ChainConsumer()
    ls = []
    for i, (chain, posterior, t, p, f, l, w, ow) in enumerate(res):
        c.add_chain(chain, posterior=posterior, walkers=l, name="Uncorrected %d"%i)
        c.add_chain(chain, weights=w, posterior=posterior, walkers=l, name="Corrected %d"%i)
        ls += ["-", "--"]
    c.configure_general(linestyles=ls)
    c.plot(filename=output, truth=t)


def plot_quick(folder, uid, include_sep=False):
    print("Performing the slowest function - quick plot. Quick to call, slow to execute.")
    td = os.path.dirname(inspect.stack()[0][1]) + "/output/"
    plot_name = td + "plot_%s.png" % uid
    plot_name_single = td + "plot_%s_single.png" % uid
    plot_name_single_weight = td + "plot_%s_single_weight.png" % uid
    plot_name_weight = td + "plot_%s_weight.png" % uid
    plot_name_sep = td + "plot_%s_sep.png" % uid
    walk_name = td + "plot_%s_walk.png" % uid

    plot_all(folder, plot_name)
    plot_all_weight(folder, plot_name_weight)
    plot_single_cosmology(folder, plot_name_single, output_walk=walk_name)
    plot_single_cosmology_weight(folder, plot_name_single_weight)
    if include_sep:
        plot_separate_weight(folder, plot_name_sep)


if __name__ == "__main__":
    dir_name = os.path.dirname(os.path.abspath(__file__))
    output = dir_name + "/output/complete.png"
    folders = ["simple", "approx"] # "stan_mc",
    use_weight = [False, True]

    c = ChainConsumer()
    for f, u in zip(folders, use_weight):
        loc = dir_name + os.sep + f + "/stan_output"
        t = None
        try:
            chain, posterior, t, p, ff, l, w, ow = load_stan_from_folder(loc, merge=True)
            if u:
                c.add_chain(chain, weights=w, posterior=posterior, walkers=l, name=f)
            else:
                c.add_chain(chain, posterior=posterior, walkers=l, name=f)
        except Exception as e:
            print(e)
            print("No files found in %s" % loc)
    c.plot(filename=output, truth=t)
