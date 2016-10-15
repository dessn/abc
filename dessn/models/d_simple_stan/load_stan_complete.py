import pickle
import os
from chainconsumer import ChainConsumer
from dessn.models.d_simple_stan.load_stan import load_stan_from_folder


def get_chain(filename, name_map):
    print("Loading chain from %s" % filename)
    with open(filename, 'rb') as output:
        chain = pickle.load(output)
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

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)

    i = 0
    td = dir_name + "/output/"
    std_simple = dir_name + "/stan_output"
    std = dir_name + "/stan_output_complete"
    chain1, posterior1, _, _, _, num_walks1 = load_stan_from_folder(std_simple)
    chain, posterior, truths, params, full_params, num_walks = load_stan_from_folder(std)
    if num_walks == 1:
        num_walks = 4
    c = ChainConsumer().add_chain(chain, posterior=posterior, walkers=num_walks)
    print(c.diagnostic_geweke())
    print(c.diagnostic_gelman_rubin())
    print("Plotting walks")
    c.plot_walks(filename=td+"complete_plot_walk.png")
    print("Plotting surfaces")
    c.plot(filename=td+"complete_plot.png", truth=truths, parameters=params)
    c.plot(filename=td+"complete_plot_full.png", truth=truths, parameters=full_params)

    c = ChainConsumer().add_chain(chain, posterior=posterior, walkers=num_walks, name="Corrected")
    c.add_chain(chain1, posterior=posterior1, walkers=num_walks1, name="Uncorrected")
    c.plot(filename=td+"complete_comparison.png", truth=truths, parameters=params)
