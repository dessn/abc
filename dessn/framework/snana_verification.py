import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

from simulations.simple import SimpleSimulation
from simulations.snana import SNANASimulationGauss0p3, SNANASimulationIdeal0p3


# Verify both simulated x1 and c and also all observed
def verify_simulation(data, alpha=0.14, beta=3.1, om=0.3, H=70, MB=-19.365, use_sim=True):
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
    if True:
        simulation = SNANASimulationIdeal0p3(-1)
        data = simulation.get_passed_supernova(-1)
        verify_simulation(data, alpha=0, beta=0)
    else:
        simulation = SimpleSimulation(40000, dscale=0, mass=False)
        data = simulation.get_passed_supernova(40000)
        verify_simulation(data, alpha=0.14, beta=3.1)
