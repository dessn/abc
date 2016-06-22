from dessn.framework.model import Model
from dessn.framework.edge import Edge
from dessn.framework.parameter import ParameterObserved, ParameterUnderlying
import sncosmo
import numpy as np


class Stretch(ParameterUnderlying):
    def __init__(self):
        super().__init__("x1", "$x_1$")

    def get_log_prior(self, data):
        if data["x1"] < -8 or data["x1"] > 8:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0

    def get_suggestion_sigma(self, data):
        return 4


class Colour(ParameterUnderlying):
    def __init__(self):
        super().__init__("c", "$c$")

    def get_suggestion_sigma(self, data):
        return 0.2

    def get_log_prior(self, data):
        if data["c"] < -1 or data["c"] > 1:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0


class Mag(ParameterUnderlying):
    def __init__(self):
        super().__init__('x0', "$x_0$")

    def get_suggestion_sigma(self, data):
        return 5e-6

    def get_log_prior(self, data):
        if data["x0"] < 0:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 1e-5


class Time(ParameterUnderlying):
    def __init__(self, t0):
        super().__init__('t0', "$t_0$")
        self.t0 = t0

    def get_suggestion_sigma(self, data):
        return 1

    def get_suggestion_requirements(self):
        return ['lc']

    def get_suggestion(self, data):
        return self.t0

    def get_log_prior(self, data):
        return 1


class Redshift(ParameterUnderlying):
    def __init__(self, zs):
        super().__init__("z", "$z$")
        self.zs = zs

    def get_suggestion_sigma(self, data):
        return 0.05

    def get_suggestion(self, data):
        return self.zs

    def get_log_prior(self, data):
        return 1


class LightCurves(ParameterObserved):
    def __init__(self, lc):
        super().__init__("lc", r"${\rm LC}$", lc)


class LikelihoodPerfect(Edge):
    def __init__(self, z):
        super().__init__(["lc"], ["x0", "x1", "c", "t0"])
        self.model = sncosmo.Model(source='salt2-extended')
        self.model.set(z=z[0])

    def get_log_likelihood(self, data):
        self.model.set(x0=data["x0"], x1=data["x1"], t0=data["t0"], c=data["c"])
        chi2 = sncosmo.chisq(data["lc"][0], self.model)
        return [-0.5 * chi2]


class LikelihoodImperfect(Edge):
    def __init__(self):
        super().__init__(["lc"], ["x0", "x1", "c", "t0", "z"])
        self.model = sncosmo.Model(source='salt2-extended')

    def get_log_likelihood(self, data):
        self.model.set(x0=data["x0"], x1=data["x1"], t0=data["t0"], c=data["c"], z=data["z"][0])
        chi2 = sncosmo.chisq(data["lc"][0], self.model)
        return [-0.5 * chi2]


class PerfectRedshift(Model):
    def __init__(self, lightcurves, redshift, t0, name="Perfect Redshift"):
        super().__init__(name)
        self.add(Stretch())
        self.add(Colour())
        self.add(Time(t0))
        self.add(Mag())
        self.add(LightCurves(lightcurves))
        self.add(LikelihoodPerfect(redshift))
        self.finalise()


class ImperfectRedshift(Model):
    def __init__(self, lightcurves, redshift, t0, name="Imperfect Redshift"):
        super().__init__(name)
        self.add(Stretch())
        self.add(Colour())
        self.add(Time(t0))
        self.add(Mag())
        self.add(Redshift(redshift))
        self.add(LightCurves(lightcurves))
        self.add(LikelihoodImperfect())
        self.finalise()
