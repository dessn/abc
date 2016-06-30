import os
import pickle
import numpy as np
from dessn.utility.generator import generate_ia_light_curve, get_summary_stats, \
    generate_ii_light_curve
import matplotlib.pyplot as plt


def fit_is_good(t0, x0, x1, c):
    return np.abs(x1) < 100 and np.abs(c) < 10 and x0 > 0 and 900 < t0 < 1100


def get_ia(z):
    mabs = np.random.normal(-19.3, 0.3)
    x1 = np.random.normal(0, 1.0)
    c = np.random.normal(0, 0.1)
    lc = generate_ia_light_curve(z, mabs, x1, c)
    try:
        res = get_summary_stats(z, lc, method="iminuit")
        assert fit_is_good(*res[0])
    except Exception:
        return None
    return res


def get_ii(z):
    mabs = np.random.normal(-18.5, 0.3)
    lc = generate_ii_light_curve(z, mabs)
    try:
        res = get_summary_stats(z, lc, method="iminuit")
        assert fit_is_good(*res[0])
    except Exception:
        return None
    return res


def get_data(seed=0, n=1000, ratio=0.8):
    np.random.seed(seed)
    ps, cs, zs, types = [], [], [], []
    while len(ps) < n:
        z = np.random.uniform(0.1, 0.9)
        if np.random.random() < ratio:
            res = get_ia(z)
            if res is not None:
                types.append(True)
                ps.append(res[0])
                cs.append(res[1])
                zs.append(z)
        else:
            res = get_ii(z)
            if res is not None:
                types.append(False)
                ps.append(res[0])
                cs.append(res[1])
                zs.append(z)
    return ps, cs, zs, types


def plot_salt2_distribution(ps, types, filename):
    iap = np.array([p for p, t in zip(ps, types) if t])
    iip = np.array([p for p, t in zip(ps, types) if not t])
    fig, ax = plt.subplots(figsize=(4, 3))
    t0, x0, x1, c = 0, 1, 2, 3
    ax.scatter(iap[:, x1], iap[:, c], c='blue', s=30, alpha=0.6, label=r"${\rm Ia}$")
    ax.scatter(iip[:, x1], iip[:, c], c='red', s=30, alpha=0.6, label=r"${\rm II}$")
    ax.legend(loc=4, frameon=False)
    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$c$", fontsize=14)
    ax.set_ylim(-1.1, 1.1)

    fig.savefig(filename, dpi=300, transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    # Set up output directory
    output_dir = os.path.dirname(__file__) + os.sep + "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data if it already exists. Generate it if it doesn't.
    pickle_file = output_dir + os.sep + "data.pkl"
    if not os.path.exists(pickle_file):
        data = get_data()
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

    ps, cs, zs, types = data
    plot_salt2_distribution(ps, types, output_dir + os.sep + "param_dist.png")
