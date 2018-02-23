import numpy as np
from scipy.stats import norm, skewnorm
import matplotlib.pyplot as plt
from scipy.integrate import simps
from matplotlib import rc


sigma = 0.1
xs = np.linspace(-5, 5, 1000)
sf = 1 - norm.cdf(xs, 2, 0.5)
mids = np.linspace(0.5, 4, 100)

alpha = 2
delta = alpha / np.sqrt(1 + alpha**2)
sigma_cor = [0.1]
shift = np.sqrt(2 / np.pi) * delta * sigma
sigma_ratio1 = np.sqrt(1 - 2 * delta**2 / np.pi)

ef_a = np.zeros(mids.shape)
ef_no = np.zeros(mids.shape)
efs = np.zeros((mids.size, len(sigma_cor)))

for i, mid in enumerate(mids):
    pop_actual = skewnorm.pdf(xs, alpha, mid, sigma)    
    pop_approx1 = norm.pdf(xs, mid, sigma)
    
    ef_a[i] = simps(pop_actual * sf, x=xs)
    ef_no[i] = simps(pop_approx1 * sf, x=xs)
    
    for j, s in enumerate(sigma_cor):
        pop = norm.pdf(xs, mid + (shift * s / sigma), sigma * sigma_ratio1)
        efs[i, j] = simps(pop * sf, x=xs)

rc('text', usetex=True)
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(4, 4), gridspec_kw={"hspace": 0.0})
axes[0].plot(mids, ef_a, label="Correct", c="k")
axes[0].plot(mids, ef_no, label="Unshifted")


axes[1].axhline(0, c="k")
axes[1].plot(mids, ef_no - ef_a, label="Unshifted")
for row, c in zip(efs.T, sigma_cor):
    axes[0].plot(mids, row, label="Shifted", ls="--")
    axes[1].plot(mids, row - ef_a, label="Shifted", ls="--")

axes[1].set_xlabel("Shift")
axes[0].set_ylabel("Efficiency")
axes[1].set_ylabel("$\Delta$ Efficiency")
axes[0].legend(frameon=False, markerfirst=False)

fig.tight_layout()
plt.savefig("shift.png", bbox_inches="tight", transparent=True)
plt.savefig("../figures/shift.pdf", bbox_inches="tight", transparent=True)