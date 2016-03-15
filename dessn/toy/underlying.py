from dessn.model.node import NodeUnderlying
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import os


class Cosmology(NodeUnderlying):
    def __init__(self):
        super(Cosmology, self).__init__("Cosmology", ["omega_m", "w", "H0"], [r"$\Omega_m$", "$w$", "$H_0$"])

    def get_log_prior(self, data):
        if data["omega_m"] < 0:
            return -np.inf
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return [0.3, -1, 70]


class SupernovaIaDist(NodeUnderlying):

    def __init__(self):
        super(SupernovaIaDist, self).__init__("SNIa", ["snIa_luminosity", "snIa_sigma"], [r"$L_{\rm SnIa}$", r"$\sigma_{\rm SnIa}$"])

    def get_log_prior(self, data):
        if data["snIa_sigma"] < 0:
            return -np.inf
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return [10, 0.5]


class SupernovaIIDist(NodeUnderlying):
    def __init__(self):
        super(SupernovaIIDist, self).__init__("SNII", ["snII_luminosity", "snII_sigma"], [r"$L_{\rm SnII}$", r"$\sigma_{\rm SnII}$"])

    def get_log_prior(self, data):
        if data["snII_sigma"] < 0:
            return -np.inf
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return [5, 0.3]


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
        r = data["sn_rate"]
        if r > 1 or r < 0:
            return -np.inf
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return [0.9]

if __name__ == "__main__":
    rate = SupernovaRate()
    rate.plot_dist()
