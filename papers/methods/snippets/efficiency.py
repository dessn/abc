import numpy as np
from scipy.stats import norm, skewnorm
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.cosmology import FlatwCDM

mbs = np.linspace(19, 27, 100)

pop = norm.pdf(mbs, 22, 1)
pop /= pop.max()

cdf_mu, cdf_sigma = 21.5, 0.5
cdf = 1 - norm.cdf(mbs, cdf_mu, cdf_sigma)
cdf_eff = pop * cdf

skew_mu, skew_sigma, skew_alpha = 20, 1, 5
skew = skewnorm.pdf(mbs, skew_alpha, skew_mu, skew_sigma)
skew /= skew.max()
skew_eff = pop * skew

rc('text', usetex=True)
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(4.5, 4.5),gridspec_kw={"hspace": 0.08})

axes[0].plot(mbs, pop, label="SNIa pop: $P(m_B|\\theta)$", ls="--")
axes[0].plot(mbs, cdf, label="CDF eff: $P(S| m_B)$", c="g", ls=":")
axes[0].plot(mbs, cdf_eff, label="$P(S| m_B)P(m_B|\\theta)$", c="g")
#axes[0].axhline(0, c='k')
axes[0].fill_between(mbs, 0, cdf_eff, color='g', alpha=0.2)
axes[0].text(20.6, 0.14, "$d_{\\rm CDF}$", fontsize=14)

axes[1].plot(mbs, pop, label="SNIa pop: $P(m_B|\\theta)$", ls="--")
axes[1].plot(mbs, skew, label="Skew eff: $P(S| m_B)$", c="r", ls=":")
axes[1].plot(mbs, skew_eff, label="$P(S| m_B)P(m_B|\\theta)$", c="r")
#axes[1].axhline(0, c='k')
axes[1].fill_between(mbs, 0, skew_eff, color='r', alpha=0.2)
axes[1].text(20.6, 0.13, "$d_{\\rm Skew}$", fontsize=14)

axes[0].legend(loc=1, markerfirst=False, frameon=False)
axes[1].legend(loc=1, markerfirst=False, frameon=False)
axes[1].set_xlabel("$m_B$")
axes[0].set_ylabel("Probability")
axes[1].set_ylabel("Probability")
axes[0].set_yticks([0, 0.5, 1])
axes[1].set_yticks([0, 0.5, 1])

axes[0].set_ylim(0, 1.1)
axes[1].set_ylim(0, 1.1)
axes[0].set_xlim(19, 27)

#fig.tight_layout()
plt.savefig("efficiency.png", dpi=300, bbox_inches="tight", transparent=True)
plt.savefig("../figures/efficiency.pdf", bbox_inches="tight", transparent=True)