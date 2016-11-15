import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

if __name__ == "__main__":
    dir_name = os.path.abspath(os.path.dirname(__file__) or ".")
    output_dir = dir_name + "/output/"
    pickle_file = output_dir + "supernovae.pickle"
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    use = [d for d in data if d['pc']]
    p = 'z'
    zs = np.array([d[p] for d in use])
    bands = [(0, '$g$', 'g'), (1, '$r$', 'r'), (2, '$i$', 'b'), (3, '$z$', 'k')]
    patches = [mpatches.Patch(color=b[2], label=b[1]) for b in bands]
    template = r"$\partial %s / \partial \mathcal{Z}_b$"
    labels = ["m_B", "x_1", "c"]
    fig, axes = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
    plt.tight_layout()

    axes[2].set_xlabel("$z$", fontsize=14)
    for i, ax in enumerate(axes):
        ax.set_ylabel(template % labels[i], fontsize=14)
        for index, label, color in bands:
            data = np.array([d['dp'][i][index] for d in use])
            ax.scatter(zs, data, lw=0, alpha=0.3, s=2, c=color, label=label)
        ax.axis('tight')
    axes[0].legend(handles=patches, frameon=False, ncol=4)
    if p == 'z':
        axes[0].set_xlim(0, 0.9)
    plt.subplots_adjust(wspace=0, hspace=0.05)

    fig.savefig(output_dir + "sensitivity.png", bbox_inches="tight", transparent=True, dpi=250)
    fig.savefig(output_dir + "sensitivity.pdf", bbox_inches="tight", transparent=True, dpi=250)