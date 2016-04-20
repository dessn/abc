from dessn.framework.parameter import ParameterUnderlying
import numpy as np


class OmegaM(ParameterUnderlying):
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

    def get_suggestion_sigma(self, data):
        return 0.05


class Hubble(ParameterUnderlying):
    def __init__(self):
        super(Hubble, self).__init__("H0", "$H_0$", group="Cosmology")
        self.sigma = 0.46 * 10
        self.prefactor = np.log(np.sqrt(2 * np.pi) * self.sigma)

    def get_log_prior(self, data):
        return -(data["H0"] - 67.74) * (data["H0"] - 67.74) / (2 * self.sigma * self.sigma) - self.prefactor

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
        return 0.1


class SupernovaIaDist2(ParameterUnderlying):

    def __init__(self):
        super(SupernovaIaDist2, self).__init__("snIa_sigma", r"$\sigma_{\rm SnIa}$", group="SNIa")

    def get_log_prior(self, data):
        if data["snIa_sigma"] < 0:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.1  # Deliberately wrong to test recovery

    def get_suggestion_sigma(self, data):
        return 0.05

