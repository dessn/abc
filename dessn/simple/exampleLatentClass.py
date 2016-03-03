import numpy as np
from dessn.simple.example import Example
from dessn.model.model import Model
from dessn.model.node import NodeObserved, NodeLatent, NodeUnderlying
from dessn.model.edge import Edge
from scipy import stats
import logging


class ObservedFlux(NodeObserved):
    def __init__(self, n=100):
        self.n = n
        flux = stats.norm.rvs(size=n, loc=100, scale=30)
        error = 0.2 * np.sqrt(flux)
        flux += stats.norm.rvs(size=n) * error

        super(ObservedFlux, self).__init__(["flux", "flux_error"], ["$f$", r"$\sigma_f$"], [flux, error])


class LatentLuminosity(NodeLatent):
    def __init__(self, n=100):
        super(LatentLuminosity, self).__init__("luminosity", "$L$")
        self.n = n

    def get_num_latent(self):
        return self.n


class UnderlyingSupernovaDistribution(NodeUnderlying):
    def get_log_prior(self, data):
        return 1

    def __init__(self):
        super(UnderlyingSupernovaDistribution, self).__init__(["SN_theta_1", "SN_theta_2"], [r"$\theta_1$", r"\theta_2"])


class FluxToLuminosity(Edge):
    def __init__(self):
        super(FluxToLuminosity, self).__init__(["flux", "flux_error"], "luminosity")

    def get_log_likelihood(self, data):
        luminosity = data["luminosity"]
        flux = data["flux"]
        flux_error = data["flux_error"]
        return -np.sum((flux - luminosity) * (flux - luminosity) / (flux_error * flux_error))


class LuminosityToSupernovaDistribution(Edge):
    def __init__(self):
        super(LuminosityToSupernovaDistribution, self).__init__("luminosity", ["SN_theta_1", "SN_theta_2"])

    def get_log_likelihood(self, data):
        luminosity = data["luminosity"]
        theta1 = data["SN_theta_1"]
        theta2 = data["SN_theta_2"]
        return -np.sum((luminosity - theta1) * (luminosity - theta1) / (theta2 * theta2))


class ExampleModel(Model):
    r"""An implementation of :class:`.ExampleLatent` using classes instead of procedural code.
    """
    def __init__(self):
        super(ExampleModel, self).__init__()

        n = 100

        flux = ObservedFlux(n=n)
        luminosity = LatentLuminosity(n=n)
        supernova = UnderlyingSupernovaDistribution()

        self.add_node(flux)
        self.add_node(luminosity)
        self.add_node(supernova)

        self.add_edge(FluxToLuminosity())
        self.add_edge(LuminosityToSupernovaDistribution())

        self.finalise()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    exampleModel = ExampleModel()
    exampleModel.fit_model()
