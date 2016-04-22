from dessn.framework.parameter import ParameterUnderlying
import numpy as np


class OmegaM(ParameterUnderlying):
    def __init__(self):
        super(OmegaM, self).__init__("omega_m", r"$\Omega_m$", group="Cosmology")

    def get_log_prior(self, data):
        om = data["omega_m"]
        if om < 0.05 or om > 0.7:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.30

    def get_suggestion_sigma(self, data):
        return 0.05


class Hubble(ParameterUnderlying):
    def __init__(self):
        super(Hubble, self).__init__("H0", "$H_0$", group="Cosmology")

    def get_log_prior(self, data):
        h0 = data["H0"]
        if h0 < 50 or h0 > 100:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 72

    def get_suggestion_sigma(self, data):
        return 5


class SupernovaIaDist1(ParameterUnderlying):

    def __init__(self):
        super(SupernovaIaDist1, self).__init__("snIa_luminosity", r"$L_{\rm SnIa}$", group="SNIa")

    def get_suggestion(self, data):
        return -19.3

    def get_suggestion_sigma(self, data):
        return 0.001


class SupernovaIaDist2(ParameterUnderlying):

    def __init__(self):
        super(SupernovaIaDist2, self).__init__("snIa_sigma", r"$\sigma_{\rm SnIa}$", group="SNIa")

    def get_log_prior(self, data):
        if data["snIa_sigma"] < 0:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.001  # Deliberately wrong to test recovery

    def get_suggestion_sigma(self, data):
        return 0.0005

