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
        return 0.2


class Hubble(ParameterUnderlying):
    def __init__(self):
        super(Hubble, self).__init__("hubble", "$H_0$", group="Cosmology")

    def get_log_prior(self, data):
        h0 = data["hubble"]
        if h0 < 50 or h0 > 100:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 72

    def get_suggestion_sigma(self, data):
        return 10


class AbsoluteMagnitude(ParameterUnderlying):

    def __init__(self):
        super(AbsoluteMagnitude, self).__init__("magnitude", r"$M_0$", group="SNIa")

    def get_suggestion_sigma(self, data):
        return 0.1

    def get_suggestion(self, data):
        return -19.3

    def get_log_prior(self, data):
        if data["magnitude"] < -21 or data["magnitude"] > -18:
            return -np.inf
        return 1


class Scatter(ParameterUnderlying):

    def __init__(self):
        super(Scatter, self).__init__("scatter", r"$\sigma_{\rm SNIa}$", group="SNIa")

    def get_log_prior(self, data):
        if data["scatter"] < 0 or data["scatter"] > 0.2:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.01  # Deliberately wrong to test recovery

    def get_suggestion_sigma(self, data):
        return 0.09


class AlphaStretch(ParameterUnderlying):
    def __init__(self):
        super().__init__("alpha", r"$\alpha$", group="Corrections")

    def get_suggestion_sigma(self, data):
        return 0.3

    def get_suggestion(self, data):
        return 0.3

    def get_log_prior(self, data):
        alpha = data["alpha"]
        if alpha < 0 or alpha > 2:
            return -np.inf
        return 1


class BetaColour(ParameterUnderlying):
    def __init__(self):
        super().__init__("beta", r"$\beta$", group="Corrections")

    def get_suggestion_sigma(self, data):
        return 3

    def get_suggestion(self, data):
        return 3

    def get_log_prior(self, data):
        beta = data["beta"]
        if beta < -1 or beta > 10:
            return -np.inf
        return -0.1 * (beta - 3) * (beta - 3)
