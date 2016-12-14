import inspect
import os
import numpy as np
from astropy.cosmology import FlatwCDM
import matplotlib.pyplot as plt

from dessn.models.d_simple_stan.run import get_analysis_data


class HubblePlotter(object):
    def __init__(self):
        self.configs = []

    def add_config(self, om, abs, adjusted_mags, zs, label, color):
        self.configs.append({"om": om, "abs": abs, "mag": adjusted_mags, "zs": zs, "label": label, "color": color})

    def plot(self, filename="hubble"):

        fig, axes = plt.subplots(nrows=2, figsize=(6, 10), gridspec_kw={"height_ratios": [2, 1]}, sharex=True)

        ax0 = axes[0]
        ax1 = axes[1]
        ax0.set_ylabel(r"$\mu$")
        ax1.set_ylabel(r"$\mu - \mu(\mathcal{C})$")
        ax1.set_xlabel("$z$")

        allz = sorted([z for entry in self.configs for z in entry["zs"]])
        zmin = np.min(allz) if len(allz) > 2 else 0
        zmax = np.max(allz) if len(allz) > 2 else 1.0
        zs = np.linspace(zmin, zmax, 100)
        fid = FlatwCDM(70, 0.3)
        fmus = fid.distmod(zs).value
        ax0.plot(zs, fmus, c='k', ls=':')
        ax1.axhline(0, c='k', ls=':')

        for config in self.configs:
            cosmo = FlatwCDM(70, config["om"])
            mus = cosmo.distmod(zs).value
            ax0.plot(zs, mus, c=config["color"], ls='--')
            ax1.plot(zs, mus - fmus, c=config["color"], ls='--')

            muc = config["mag"] - config["abs"]
            ax0.scatter(config["zs"], muc, label=config["label"], lw=0, s=6, c=config["color"], alpha=0.3)
            ax1.scatter(config["zs"], muc - fid.distmod(config["zs"]).value, s=6, lw=0, c=config["color"], alpha=0.3)

        ax0.legend(loc=2)
        plt.subplots_adjust(wspace=0, hspace=0.05)
        this_file = inspect.stack()[0][1]
        dir_name = os.path.dirname(this_file)
        output_dir = dir_name + "/output/"
        print("Saving to " + output_dir + "%s.png" % filename)
        fig.savefig(output_dir + "%s.png" % filename, bbox_inches="tight", transparent=True, dpi=250)
        fig.savefig(output_dir + "%s.pdf" % filename, bbox_inches="tight", transparent=True, dpi=250)

if __name__ == "__main__":
    h = HubblePlotter()

    data = get_analysis_data(seed=0)
    zs = data["redshifts"]
    deta_dcalib = data["deta_dcalib"]
    obs_mBx1c = np.array(data["obs_mBx1c"])
    abs = -19.365
    mags = obs_mBx1c[:, 0] + 0.14 * obs_mBx1c[:, 1] - 3.1 * obs_mBx1c[:, 2]
    obs_mBx1c2 = np.copy(obs_mBx1c)
    calib = np.array([0, 0.01*9, 0, 0])
    calib_change = np.dot(deta_dcalib, calib)
    obs_mBx1c2 += calib_change
    mags2 = obs_mBx1c2[:, 0] + 0.14 * obs_mBx1c2[:, 1] - 3.1 * obs_mBx1c2[:, 2]

    h.add_config(0.3, abs, mags, zs, "No calib", 'b')
    h.add_config(0.24, abs, mags2, zs, "calib", 'r')

    h.plot()
