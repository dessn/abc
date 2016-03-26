from dessn.model.node import NodeUnderlying
import numpy as np
import os


class OmegaM(NodeUnderlying):
    def __init__(self):
        super(OmegaM, self).__init__("omega_m", r"$\Omega_m$", group="Cosmology")
        self.sigma = 0.0062 * 10
        self.prefactor = np.log(np.sqrt(2 * np.pi) * self.sigma) # Planck 2016

    def get_log_prior(self, data):
        om = data["omega_m"]
        if om < 0.05 or om > 0.7:
            return -np.inf
        return -(om - 0.3089) * (om - 0.3089) / (2 * self.sigma * self.sigma) - self.prefactor

    def get_suggestion(self, data):
        return 0.30


class W(NodeUnderlying):
    def __init__(self):
        super(W, self).__init__("w", "$w$", group="Cosmology")
        self.sigma = 0.08 * 10
        self.prefactor = np.log(np.sqrt(2 * np.pi) * self.sigma)

    def get_log_prior(self, data):
        return -(data["w"] - 1.019) * (data["w"]* - 1.019) / (2 * self.sigma * self.sigma) - self.prefactor

    def get_suggestion(self, data):
        return -1.0


class Hubble(NodeUnderlying):
    def __init__(self):
        super(Hubble, self).__init__("H0", "$H_0$", group="Cosmology")
        self.sigma = 0.46 * 10
        self.prefactor = np.log(np.sqrt(2 * np.pi) * self.sigma)

    def get_log_prior(self, data):
        return -(data["H0"] - 67.74) * (data["H0"] - 67.74) / (2 * self.sigma * self.sigma) - self.prefactor

    def get_suggestion(self, data):
        return 72


class SupernovaIaDist1(NodeUnderlying):

    def __init__(self):
        super(SupernovaIaDist1, self).__init__("snIa_luminosity", r"$L_{\rm SnIa}$", group="SNIa")

    def get_suggestion(self, data):
        return 10


class SupernovaIaDist2(NodeUnderlying):

    def __init__(self):
        super(SupernovaIaDist2, self).__init__("snIa_sigma", r"$\sigma_{\rm SnIa}$", group="SNIa")

    def get_log_prior(self, data):
        if data["snIa_sigma"] < 0:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.1


class SupernovaIIDist1(NodeUnderlying):
    def __init__(self):
        super(SupernovaIIDist1, self).__init__("snII_luminosity", r"$L_{\rm SnII}$", group="SNII")

    def get_suggestion(self, data):
        return 9.9


class SupernovaIIDist2(NodeUnderlying):
    def __init__(self):
        super(SupernovaIIDist2, self).__init__("snII_sigma", r"$\sigma_{\rm SnII}$", group="SNII")

    def get_log_prior(self, data):
        if data["snII_sigma"] < 0:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.02


class SupernovaRate(NodeUnderlying):
    def __init__(self):
        super(SupernovaRate, self).__init__("sn_rate", "$r$", group="SN Rates")

    def get_log_prior(self, data):
        r = data["sn_rate"]
        if r > 1 or r < 0:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.6
