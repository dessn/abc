from dessn.model.node import NodeUnderlying
import numpy as np
import os


class OmegaM(NodeUnderlying):
    def __init__(self):
        super(OmegaM, self).__init__("omega_m", r"$\Omega_m$", group="Cosmology")

    def get_log_prior(self, data):
        if data["omega_m"] < 0:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.3


class W(NodeUnderlying):
    def __init__(self):
        super(W, self).__init__("w", "$w$", group="Cosmology")

    def get_suggestion(self, data):
        return -1


class Hubble(NodeUnderlying):
    def __init__(self):
        super(Hubble, self).__init__("H0", "$H_0$", group="Cosmology")

    def get_suggestion(self, data):
        return 70


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
        return 0.5


class SupernovaIIDist1(NodeUnderlying):
    def __init__(self):
        super(SupernovaIIDist1, self).__init__("snII_luminosity", r"$L_{\rm SnII}$", group="SNII")

    def get_suggestion(self, data):
        return 5


class SupernovaIIDist2(NodeUnderlying):
    def __init__(self):
        super(SupernovaIIDist2, self).__init__("snII_sigma", r"$\sigma_{\rm SnII}$", group="SNII")

    def get_log_prior(self, data):
        if data["snII_sigma"] < 0:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.3


class SupernovaRate(NodeUnderlying):
    def __init__(self):
        super(SupernovaRate, self).__init__("sn_rate", "$r$", group="SN Rates")

    def get_log_prior(self, data):
        r = data["sn_rate"]
        if r > 1 or r < 0:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.8
