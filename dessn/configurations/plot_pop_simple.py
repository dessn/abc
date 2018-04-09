import os

from mpl_toolkits.axes_grid1 import make_axes_locatable

from dessn.framework.simulations.simple import SimpleSimulation
from astropy.cosmology import FlatwCDM
import matplotlib.pyplot as plt
from matplotlib import rc, ticker

if __name__ == "__main__":

    plot_dir = os.path.dirname(os.path.abspath(__file__)) + "/plots/%s/" % os.path.basename(__file__)[:-3]
    pfn = plot_dir + os.path.basename(__file__)[:-3]
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    sims = [SimpleSimulation(1000), SimpleSimulation(1000, lowz=True)]
    names = ["High-redshift", "Low-redshift"]

    cosmo = FlatwCDM(70, 0.3)
    rc('text', usetex=True)
    fig, axes = plt.subplots(nrows=len(sims), figsize=(4.5, 5))

    for ax, sim, n in zip(axes, sims, names):
        # ax, ax2 = axs
        data = sim.get_all_supernova(1000)
        obs_mBx1c = data["obs_mBx1c"]
        mbs, x1s, cs = obs_mBx1c[:, 0], obs_mBx1c[:, 1], obs_mBx1c[:, 2]
        mbs, x1s, cs = data["sim_apparents"], data["sim_stretches"], data["sim_colours"]
        zs = data["redshifts"]
        passed = data["passed"]
        mus = cosmo.distmod(zs).value
        MBs = mbs - mus
        h1 = ax.scatter(zs[passed], MBs[passed], c=cs[passed], s=10, vmin=-0.2, vmax=0.4, alpha=1, cmap="plasma")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        fig.colorbar(h1, cax=cax, orientation='vertical', label="Colour")
        # ax2.scatter(zs[passed], cs[passed], c=cs[passed], s=10, vmin=-0.2, vmax=0.4, alpha=1, cmap="plasma")

        ax.set_xlabel("$z$", fontsize=14)
        ax.set_ylabel(r"$m_B - \mu$", fontsize=14)
        ax.text(0.98, 0.95, n, verticalalignment='top', horizontalalignment='right', transform=ax.transAxes)
        ax.set_xlim(zs.min() - 0.001, zs.max() + 0.001)
        # ax2.set_xlabel("$z$", fontsize=14)
        # ax2.set_ylabel(r"$c$", fontsize=14)
        # ax2.text(0.98, 0.95, n, verticalalignment='top', horizontalalignment='right', transform=ax.transAxes)
        # ax2.set_xlim(zs.min() - 0.001, zs.max() + 0.001)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))

    fig.tight_layout()

    fig.savefig(pfn + ".png", dpi=200, bbox_inches="tight", transparent=True)
    fig.savefig(pfn + ".pdf", bbox_inches="tight", transparent=True)
