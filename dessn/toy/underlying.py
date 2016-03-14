from dessn.model.node import NodeUnderlying
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import os



class Cosmology(NodeUnderlying):
    def __init__(self):
        super(Cosmology, self).__init__("Cosmology", ["omega_m", "w", "H0"], [r"$\Omega_m$", "$w$", "$H_0$"])

    def get_log_prior(self, data):
        return 1


class SupernovaIaDist(NodeUnderlying):
    def __init__(self):
        super(SupernovaIaDist, self).__init__("SNIa", ["snIa_luminosity", "snIa_sigma"], ["$L$", r"$\sigma_L$"])

    def get_log_prior(self, data):
        return 1


class SupernovaIIDist(NodeUnderlying):
    def __init__(self):
        super(SupernovaIIDist, self).__init__("SNII", ["snII_luminosity", "snII_sigma"], ["$L$", r"$\sigma_L$"])

    def get_log_prior(self, data):
        return 1


class SupernovaRate(NodeUnderlying):
    def __init__(self):
        super(SupernovaRate, self).__init__("SN Rates", "sn_rate", "$r$")
        self.b = beta(30, 2)

    def plot_dist(self):
        """ Plots the distribution for easy visualisation.
        """
        f = os.path.dirname(__file__) + "/../../plots/SupernovaRateDist.png"

        xs = np.linspace(0, 1, 1000)
        ys = self.b.pdf(xs)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(xs, ys)
        ax.set_xlabel("$r$")
        ax.set_ylabel("$P(r)$")
        fig.savefig(f, dpi=300, transparent=True, bbox_inches="tight")

    def get_log_prior(self, data):
        r""" Here we model the prior on the supernova rate as a Beta function,
        parametrised as :math:`{\rm Beta}(\alpha=30, \beta=2)`.

        The probability density function is visualised below, where :math:`r` represents
        the overall ratio of type Ia supernova over all supernova.

        .. figure::     ../plots/SupernovaRateDist.png
            :align:     center
        """
        r = data["sn_rate"]
        return np.log(self.b.pdf(r))


if __name__ == "__main__":
    rate = SupernovaRate()
    rate.plot_dist()
