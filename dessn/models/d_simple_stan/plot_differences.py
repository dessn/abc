import inspect
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


if __name__ == "__main__":
    filename = "shifts"
    print("Getting data from supernovae pickle")
    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    pickle_file = os.path.abspath(this_dir + "/output/supernovae.pickle")
    with open(pickle_file, 'rb') as pkl:
        supernovae = pickle.load(pkl)

    passed = [s for s in supernovae if s["pc"] and s["z"] < 0.4]
    mb1 = np.array([s["mB"] for s in passed])
    c1 = np.array([s["c"] for s in passed])
    x11 = np.array([s["x1"] for s in passed])
    mb2 = np.array([s["parameters"][0] for s in passed])
    x12 = np.array([s["parameters"][1] for s in passed])
    c2 = np.array([s["parameters"][2] for s in passed])

    diff_mb = mb1 - mb2
    diff_c = c1 - c2
    diff_x1 = x11 - x12
    zs = np.array([s["z"] for s in passed])

    fig, axes = plt.subplots(nrows=3, figsize=(6, 6), sharex=True)

    for ax, p in zip(axes, [diff_mb, diff_x1, diff_c]):
        ax.scatter(zs, p, s=3, lw=0.1, alpha=0.5)
        ax.axhline(0, c='k')

    this_file = inspect.stack()[0][1]
    dir_name = os.path.dirname(this_file)
    output_dir = dir_name + "/output/"
    print("Saving to " + output_dir + "%s.png" % filename)
    fig.savefig(output_dir + "%s.png" % filename, bbox_inches="tight", transparent=True, dpi=250)
    fig.savefig(output_dir + "%s.pdf" % filename, bbox_inches="tight", transparent=True, dpi=250)

