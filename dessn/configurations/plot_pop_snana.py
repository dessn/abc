import os

from dessn.framework.simulations.simple import SimpleSimulation
from astropy.cosmology import FlatwCDM
import matplotlib.pyplot as plt
from matplotlib import rc

from dessn.framework.simulations.snana import SNANASimulation

if __name__ == "__main__":

    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    pfn = plot_dir + os.path.basename(__file__)[:-3]
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    sims = [SNANASimulation(-1, "DES3YR_DES_BULK_G10_SKEW"), SNANASimulation(-1, "DES3YR_LOWZ_BULK_G10_SKEW")]
    names = ["High-redshift", "Low-redshift"]

    cosmo = FlatwCDM(70, 0.3)
    rc('text', usetex=True)
    fig, axes = plt.subplots(nrows=len(sims), figsize=(4, 5))

    for ax, sim, n in zip(axes, sims, names):
        data = sim.get_passed_supernova(-1)
        obs_mBx1c = data["obs_mBx1c"]
        mbs, x1s, cs = obs_mBx1c[:, 0], obs_mBx1c[:, 1], obs_mBx1c[:, 2]
        mbs, x1s, cs = data["sim_apparents"], data["sim_stretches"], data["sim_colours"]
        # print(cs)
        # exit()
        zs = data["redshifts"]
        mus = cosmo.distmod(zs).value
        MBs = mbs - mus
        ax.scatter(zs, MBs, c=cs, s=10, vmin=-0.2, vmax=0.4, alpha=1, cmap="plasma")

        ax.set_xlabel("$z$", fontsize=14)
        ax.set_ylabel(r"$m_B - \mu$", fontsize=14)
        ax.text(0.98, 0.95, n, verticalalignment='top', horizontalalignment='right', transform=ax.transAxes)
        ax.set_xlim(zs.min() - 0.001, zs.max() + 0.001)
    plt.locator_params(numticks=5)
    fig.tight_layout()

    fig.savefig(pfn + ".png", bbox_inches="tight", transparent=True)
    fig.savefig(pfn + ".pdf", bbox_inches="tight", transparent=True)
