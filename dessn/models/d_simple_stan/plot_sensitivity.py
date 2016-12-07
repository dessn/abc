import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from astropy.io import fits

if __name__ == "__main__":
    dir_name = os.path.abspath(os.path.dirname(__file__) or ".")
    output_dir = dir_name + "/output/"
    data_dir = dir_name + "/data/"
    pickle_file = data_dir + "supernovae.pickle"
    chris_file = data_dir + "J_20161120.fits"
    chris_file2 = data_dir + "zs.txt"
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    f = fits.open(chris_file)
    datac = f[0].data
    f.close()
    c_dm = datac[::3, :] * 100
    c_dx1 = datac[1::3, :] * 100
    c_dc = datac[2::3, :] * 100
    c_data = [c_dm, c_dx1, c_dc]
    c_zs = np.loadtxt(chris_file2)
    c_colors = ['#0D6B18', '#A30505', '#131787', '#7D7D7D']
    plot_chris = True

    use = [d for d in data if d['pc']]
    p = 'z'
    zs = np.array([d[p] for d in use])
    bands = [(0, '$g$', 'g'), (1, '$r$', 'r'), (2, '$i$', 'b'), (3, '$z$', 'k')]
    patches = [mpatches.Patch(color=b[2], label=b[1]) for b in bands]
    template = r"$\partial %s / \partial \mathcal{Z}_b$"
    labels = ["m_B", "x_1", "c"]
    fig, axes = plt.subplots(4, 1, figsize=(5, 8), sharex=True)
    plt.tight_layout()

    axes[3].set_xlabel("$z$", fontsize=14)
    for i, ax in enumerate(axes[:-1]):
        ax.set_ylabel(template % labels[i], fontsize=14)
        for index, label, color in bands:
            data = np.array([d['dp'][i][index] for d in use])
            ax.scatter(zs, data, lw=0, alpha=0.3, s=2, c=color, label=label)
        if plot_chris:
            for col, c in zip(range(c_data[i].shape[1]), c_colors):
                ax.scatter(c_zs, c_data[i][:, col], lw=0, alpha=0.2, s=2, c=c)
        ax.axis('tight')

    ax = axes[-1]
    ax.set_ylabel(r"$\partial m_B^* / \partial \mathcal{Z}_b, \ \alpha=0.1,\  \beta=3$")
    for (index, label, color), c2 in zip(bands, c_colors):
        data = np.array([d['dp'][0][index]+0.1*d['dp'][1][index] - 3.0*d['dp'][2][index] for d in use])
        ax.scatter(zs, data, lw=0, alpha=0.3, s=2, c=color, label=label)
        if plot_chris:
            data2 = c_dm[:, index] + 0.1 * c_dx1[:, index] - 3.0 * c_dc[:, index]
            ax.scatter(c_zs, data2, lw=0, alpha=0.2, s=2, c=c2)

    axes[0].legend(handles=patches, frameon=False, ncol=4)
    if p == 'z':
        axes[0].set_xlim(0, 1.0)
    axes[0].set_ylim(-1.2, 0.8)
    plt.subplots_adjust(wspace=0, hspace=0.05)
    print("Saving to " + output_dir + "sensitivity.png")
    fig.savefig(output_dir + "sensitivity.png", bbox_inches="tight", transparent=True, dpi=250)
    fig.savefig(output_dir + "sensitivity.pdf", bbox_inches="tight", transparent=True, dpi=250)