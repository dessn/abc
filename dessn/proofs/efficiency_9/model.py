from dessn.framework.model import Model
from dessn.framework.edge import Edge
from dessn.framework.parameter import ParameterUnderlying, ParameterObserved, ParameterLatent
import numpy as np

# Observed Parameters
###############################################################################

class ObservedRedshift(ParameterObserved):
    def __init__(self, zs):
        super().__init__("z", "$z$", zs)


class ObservedType(ParameterObserved):
    def __init__(self, types):
        super().__init__("t", "$T$", types)


class ObservedSummary(ParameterObserved):
    def __init__(self, ps):
        super().__init__("params", r"$t_0,\,x_0,\,x_1,\,c$", ps, group="Salt2")


class ObservedCov(ParameterObserved):
    def __init__(self, cs):
        super().__init__("covs", r"$\hat{C}$", cs, group="Salt2")


# Underlying Parameters
###############################################################################

class IaAlpha(ParameterUnderlying):
    def __init__(self):
        super().__init__("IaAlpha", r"$\alpha$", group="Ia")

    def get_suggestion_sigma(self, data):
        return 0.5

    def get_suggestion(self, data):
        return 0

    def get_log_prior(self, data):
        if -2 < data["IaAlpha"] < 2:
            return -np.inf
        return 1


class IIAlpha(ParameterUnderlying):
    def __init__(self):
        super().__init__("IIAlpha", r"$\alpha$", group="II")

    def get_suggestion_sigma(self, data):
        return 0.5

    def get_suggestion(self, data):
        return 0

    def get_log_prior(self, data):
        if -2 < data["IaAlpha"] < 2:
            return -np.inf
        return 1


class IaBeta(ParameterUnderlying):
    def __init__(self):
        super().__init__("IaBeta", r"$\beta$", group="Ia")

    def get_suggestion_sigma(self, data):
        return 0.5

    def get_suggestion(self, data):
        return 0

    def get_log_prior(self, data):
        if -2 < data["IaAlpha"] < 2:
            return -np.inf
        return 1


class IIBeta(ParameterUnderlying):
    def __init__(self):
        super().__init__("IIBeta", r"$\beta$", group="II")

    def get_suggestion_sigma(self, data):
        return 0.5

    def get_suggestion(self, data):
        return 0

    def get_log_prior(self, data):
        if -2 < data["IaAlpha"] < 2:
            return -np.inf
        return 1


class IaMagnitude(ParameterUnderlying):
    def __init__(self):
        super().__init__("IaMag", r"$M$", group="Ia")

    def get_suggestion_sigma(self, data):
        return 1

    def get_suggestion(self, data):
        return -19

    def get_log_prior(self, data):
        if -21 < data["IaMag"] < -18:
            return -np.inf
        return 1


class IIMagnitude(ParameterUnderlying):
    def __init__(self):
        super().__init__("IIMag", r"$M$", group="IO")

    def get_suggestion_sigma(self, data):
        return 1

    def get_suggestion(self, data):
        return -18

    def get_log_prior(self, data):
        if -20 < data["IIMag"] < -16:
            return -np.inf
        return 1


class TwoPopulationModel(Model):
    def __init__(self, zs, types, ps, cs, name="TwoPopulationModel"):
        super().__init__(name)

        # Add observed quantities
        self.add(ObservedRedshift(zs))
        self.add(ObservedType(types))
        self.add(ObservedSummary(ps))
        self.add(ObservedCov(cs))

        # Add underlying parameters
        self.add(IaAlpha())
        self.add(IIAlpha())
        self.add(IaBeta())
        self.add(IIBeta())
        self.add(IaMagnitude())
        self.add(IIMagnitude())

