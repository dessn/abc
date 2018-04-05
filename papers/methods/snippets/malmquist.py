import numpy as np
from scipy.stats import norm, skewnorm
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.cosmology import FlatwCDM

cosmo = FlatwCDM(70, 0.3)

zs = np.linspace(0.1, 1.0, 100)
mus = cosmo.distmod(zs).value

MB = -19.3
alpha = 0.1
beta = 3.1

n = 1000
np.random.seed(5)
sn_zs = np.random.uniform(0.1, 1.0, size=n)
sn_mus = cosmo.distmod(sn_zs).value
sn_MBs = np.random.normal(MB, 0.2, size=n)
sn_cs = np.random.normal(0, 0.04, size=n)
sn_x1s = np.random.normal(0, 0.5, size=n)
sn_mbs = sn_MBs - alpha * sn_x1s + beta * sn_cs + sn_mus

sn_disp = sn_mbs + alpha * sn_x1s - beta * sn_cs - MB
#mask = np.random.uniform(size=n) <= (1 - norm.cdf(sn_mbs, 24, 0.5))
mask = sn_mbs < 24

from scipy.stats import binned_statistic
means, bins, _ = binned_statistic(sn_zs[mask], sn_disp[mask])
err, bins, _ = binned_statistic(sn_zs[mask], sn_disp[mask], bins=bins, statistic=lambda x: np.std(x)/np.sqrt(len(x)))
bin_center = 0.5 * (bins[1:] + bins[:-1])
bias = means - cosmo.distmod(bin_center).value

#rc('text', usetex=True)
fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(4.5, 3.5), gridspec_kw={"hspace": 0.0})

ax.plot(zs, mus, c='k', ls='--')
ax.scatter(sn_zs[mask], sn_disp[mask], s=2, c="#4286f4", alpha=0.9, label="Observed SN")
ax.scatter(sn_zs[~mask], sn_disp[~mask], s=2, c="#f44141", alpha=0.4, label="Unobserved SN")


cax = ax.scatter(bin_center, means, c=np.abs(bias), s=50, lw=1, edgecolor='k', label="SN means")
cbar = fig.colorbar(cax)
cbar.set_label(r"$\Delta \mu$", fontsize=14)

ax.set_xlabel("$z$", fontsize=14)
ax.set_ylabel(r"$\mu$", fontsize=14)

ax.legend(loc=4, frameon=False, markerfirst=False)

fig.tight_layout()
plt.savefig("malmquist.png", bbox_inches="tight", transparent=True)
plt.savefig("../figures/malmquist.pdf", bbox_inches="tight", transparent=True)