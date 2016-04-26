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


class Magnitude(ParameterUnderlying):
    def __init__(self):
        super(Magnitude, self).__init__("mag", r"$M_B$", group="SNIa")

    def get_suggestion(self, data):
        return -19.3

    def get_suggestion_sigma(self, data):
        return 10


class IntrinsicScatter(ParameterUnderlying):
    def __init__(self):
        super(IntrinsicScatter, self).__init__("scatter", r"$\sigma_{\rm int}$", group="SNIa")

    def get_log_prior(self, data):
        if data["scatter"] < 0:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.1  # Deliberately wrong to test recovery

    def get_suggestion_sigma(self, data):
        return 0.08


class AlphaStretch(ParameterUnderlying):
    def __init__(self):
        super().__init__("alpha", r"$\alpha$", group="Corrections")

    def get_suggestion_sigma(self, data):
        return 0.5

    def get_suggestion(self, data):
        return 0.1


class BetaColour(ParameterUnderlying):
    def __init__(self):
        super().__init__("beta", r"$\beta$", group="Corrections")

    def get_suggestion_sigma(self, data):
        return 2

    def get_suggestion(self, data):
        return 3
