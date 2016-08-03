from dessn.framework.model import Model
from dessn.framework.edge import Edge
from dessn.framework.parameter import ParameterObserved, ParameterUnderlying
import numpy as np
from astropy.cosmology import FlatwCDM


class Redshifts(ParameterObserved):
    def __init__(self, zs):
        super().__init__("z", "$z$", zs)


class Mus(ParameterObserved):
    def __init__(self, mus):
        super().__init__("mu", r"$\mu$", mus)


class MuError(ParameterObserved):
    def __init__(self, mus):
        super().__init__("mue", r"$\sigma_{\mu}$", mus)


class M(ParameterUnderlying):
    def __init__(self):
        super().__init__("M", r"$M$")

    def get_suggestion(self, data):
        return -19.3

    def get_suggestion_sigma(self, data):
        return 0.5

    def get_log_prior(self, data):
        om = data["M"]
        if om < -20 or om > -19:
            return -np.inf
        return 1


class OmegaM(ParameterUnderlying):
    def __init__(self):
        super().__init__("omega_m", r"$\Omega_m$")

    def get_suggestion(self, data):
        return 0.3

    def get_suggestion_sigma(self, data):
        return 0.1

    def get_log_prior(self, data):
        om = data["omega_m"]
        if om < 0.1 or om > 0.6:
            return -np.inf
        return 1


class W(ParameterUnderlying):
    def __init__(self):
        super().__init__("w", r"$w$")

    def get_suggestion(self, data):
        return -1

    def get_suggestion_sigma(self, data):
        return 0.3

    def get_log_prior(self, data):
        om = data["w"]
        if om < -2 or om > -0.2:
            return -np.inf
        return 1


class BiasCorrection(Edge):
    def __init__(self, interpolator):
        super().__init__("z", ["omega_m", "w", "M"])
        self.interpolator = interpolator

    def get_log_likelihood(self, data):
        omega_m, w, mabs = data["omega_m"], data["w"], data["M"]
        biases = self.interpolator([omega_m, w, mabs])
        ps = -np.log(biases)
        ps[ps == np.inf] = -np.inf
        return np.ones(data["z"].shape)# * ps


class Likelihood(Edge):
    def __init__(self):
        self.H0 = 70
        super().__init__(["z", "mu", "mue"], ["omega_m", "w", "M"])
        # super().__init__(["z", "mu", "mue"], ["omega_m", "w"])

    def get_log_likelihood(self, data):
        cosmology = FlatwCDM(H0=self.H0, Om0=data["omega_m"], w0=data["w"])
        distmod = cosmology.distmod(data["z"]).value
        error = np.sqrt(data["mue"]**2 + 0.05*0.05)
        diff = (distmod - data["mu"] + data["M"]) / error
        # diff = (distmod - data["mu"]) / data["mue"]
        return -0.5 * (diff * diff)


class SimpleCosmologyFitter(Model):
    def __init__(self, name, zs, mus, mues, interpolator):
        super().__init__(name)
        self.add(Redshifts(zs))
        self.add(Mus(mus))
        self.add(MuError(mues))
        self.add(OmegaM())
        self.add(W())
        self.add(M())
        self.add(Likelihood())
        self.add(BiasCorrection(interpolator))
        self.finalise()