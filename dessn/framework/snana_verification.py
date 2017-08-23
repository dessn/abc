import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

from simulations.simple import SimpleSimulation
from simulations.snana import SNANASimulationGauss0p3, SNANASimulationIdeal0p3, SNANASimulationIdealNoBias0p3, \
    SNANASimulationLowzGauss0p3


def verify_simulation(simulation, alpha=0.14, beta=3.1, om=0.3, H=70, MB=-19.365):

    # data_passed = simulation.get_passed_supernova(-1)
    data_all = simulation.get_all_supernova(-1)

    bin_count = 100
    name = simulation.__class__.__name__
    print(data_all.keys())
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axes = axes.flatten()

    cor = simulation.get_approximate_correction()

    fig.suptitle("%s   %0.3f %0.3f %s" % (name, cor[0], cor[1], cor[2:]), y=0.997)

    ax = axes[0]
    mask = data_all["passed"]
    all_mb, bins = np.histogram(data_all["sim_apparents"], bins=bin_count)
    passed_mb, _ = np.histogram(data_all["sim_apparents"][mask], bins=bins)
    ratio = passed_mb / (all_mb + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    ax.plot(bc, ratio, label="Ratio passed")
    ax.set_xlabel("mB_sim")
    ax.set_ylabel("P")
    ax.legend()

    ax = axes[1]
    ax.plot(bc, np.log10(all_mb), label="All mB")
    ax.plot(bc, np.log10(passed_mb), label="Passed mB")
    ax.set_xlabel("mB_sim")
    ax.set_ylabel("log10 N")
    ax.legend()

    ax = axes[2]
    mask = data_all["passed"]
    all_mb, bins = np.histogram(data_all["redshifts"], bins=bin_count)
    passed_mb, _ = np.histogram(data_all["redshifts"][mask], bins=bins)
    ratio = passed_mb / (all_mb + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    ax.plot(bc, ratio, label="Ratio passed")
    ax.set_xlabel("z")
    ax.set_ylabel("P")
    ax.legend()

    ax = axes[3]
    ax.plot(bc, np.log10(all_mb), label="All z")
    ax.plot(bc, np.log10(passed_mb), label="Passed z")
    ax.set_xlabel("z")
    ax.set_ylabel("log10 N")
    ax.legend()

    fig.tight_layout()
    fig.savefig(name + ".png")
    plt.show()

# Verify both simulated x1 and c and also all observed
def verify_simulation2(data, alpha=0.14, beta=3.1, om=0.3, H=70, MB=-19.365, use_sim=True):
    zs = data['redshifts']
    print("Data size is ", zs.shape)

    if use_sim:
        mbs = data['sim_apparents']
        x1s = data['sim_stretches']
        cs = data['sim_colours']
    else:
        mbs = data["obs_mBx1c"][:, 0]
        x1s = data["obs_mBx1c"][:, 1]
        cs = data["obs_mBx1c"][:, 2]
    mus = FlatLambdaCDM(H, om).distmod(zs).value
    MBs = mbs - mus + alpha * x1s - beta * cs
    diff = MBs - MB

    means, be, bn = binned_statistic(zs, diff)
    bc = 0.5 * (be[1:] + be[:-1])
    print(be)
    print(bc)
    plt.plot(zs, diff, 'bo', markeredgecolor='none', markersize=5, alpha=0.03)
    plt.plot(bc, means, 'rs', markersize=5)
    plt.axhline(0, c='k', ls='--')
    plt.xlabel("$z$")
    plt.ylabel("$\Delta M_B$")
    plt.show()

    print(MBs.mean(), np.std(MBs))


if __name__ == "__main__":
    # verify_simulation(SNANASimulationIdeal0p3(-1), alpha=0, beta=0)
    # verify_simulation(SNANASimulationIdealNoBias0p3(-1), alpha=0, beta=0)
    # verify_simulation(SNANASimulationGauss0p3(-1), alpha=0.14, beta=3.1)
    verify_simulation(SNANASimulationLowzGauss0p3(-1), alpha=0.14, beta=3.1)
