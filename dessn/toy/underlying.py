from dessn.model.parameter import ParameterUnderlying
import numpy as np


class ZCalibration(ParameterUnderlying):
    def __init__(self):
        super(ZCalibration, self).__init__("Zcal", "$Z$", group="Calibration")

    def get_log_prior(self, data):
        z = data["Zcal"]
        if z < 0 or z > 10:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 6

    def get_suggestion_sigma(self, data):
        return 1


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


class W(ParameterUnderlying):
    def __init__(self):
        super(W, self).__init__("w", "$w$", group="Cosmology")
        self.sigma = 0.08 * 10
        self.prefactor = np.log(np.sqrt(2 * np.pi) * self.sigma)

    def get_log_prior(self, data):
        return -(data["w"] - 1.019) * (data["w"]* - 1.019) / (2 * self.sigma * self.sigma) - self.prefactor

    def get_suggestion(self, data):
        return -1.0

    def get_suggestion_sigma(self, data):
        return 0.03


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
        return 10

    def get_suggestion_sigma(self, data):
        return 0.05


class SupernovaIaDist2(ParameterUnderlying):

    def __init__(self):
        super(SupernovaIaDist2, self).__init__("snIa_sigma", r"$\sigma_{\rm SnIa}$", group="SNIa")

    def get_log_prior(self, data):
        if data["snIa_sigma"] < 0:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.03  # Deliberately wrong to test recovery

    def get_suggestion_sigma(self, data):
        return 0.02


class SupernovaIIDist1(ParameterUnderlying):
    def __init__(self):
        super(SupernovaIIDist1, self).__init__("snII_luminosity", r"$L_{\rm SnII}$", group="SNII")

    def get_suggestion(self, data):
        return 9.8

    def get_suggestion_sigma(self, data):
        return 0.05


class SupernovaIIDist2(ParameterUnderlying):
    def __init__(self):
        super(SupernovaIIDist2, self).__init__("snII_sigma", r"$\sigma_{\rm SnII}$", group="SNII")

    def get_log_prior(self, data):
        if data["snII_sigma"] < 0:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.02

    def get_suggestion_sigma(self, data):
        return 0.01


class SupernovaRate(ParameterUnderlying):
    def __init__(self):
        super(SupernovaRate, self).__init__("sn_rate", "$r$", group="SN Rates")

    def get_log_prior(self, data):
        r = data["sn_rate"]
        if r > 1 or r < 0:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.6

    def get_suggestion_sigma(self, data):
        return 0.2
