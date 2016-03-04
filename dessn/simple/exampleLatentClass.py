import numpy as np
from dessn.model.model import Model
from dessn.model.node import NodeObserved, NodeLatent, NodeUnderlying, NodeTransformation
from dessn.model.edge import Edge, EdgeTransformation
from scipy import stats
import logging


class ObservedFlux(NodeObserved):
    def __init__(self, n=100):
        self.n = n
        flux = stats.norm.rvs(size=n, loc=100, scale=20) / 2
        error = 0.001 * np.sqrt(flux)
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
        super(UnderlyingSupernovaDistribution, self).__init__(["SN_theta_1", "SN_theta_2"],
                                                              [r"$\theta_1$", r"$\theta_2$"])


class UselessTransformation(NodeTransformation):
    def __init__(self):
        super(UselessTransformation, self).__init__("double_luminosity", "$2L$")


class LuminosityToAdjusted(EdgeTransformation):
    def __init__(self):
        super(LuminosityToAdjusted, self).__init__("luminosity", "double_luminosity")

    def get_transformation(self, data):
        return {"double_luminosity": data["luminosity"] * 2.0}


class FluxToLuminosity(Edge):
    def __init__(self):
        super(FluxToLuminosity, self).__init__(["flux", "flux_error"], "luminosity")

    def get_log_likelihood(self, data):
        luminosity = data["luminosity"]
        flux = data["flux"]
        flux_error = data["flux_error"]
        return -np.sum((flux - luminosity) * (flux - luminosity) / (flux_error * flux_error) + np.log(np.sqrt(2 * np.pi) * flux_error))


class LuminosityToSupernovaDistribution(Edge):
    def __init__(self):
        super(LuminosityToSupernovaDistribution, self).__init__("double_luminosity", ["SN_theta_1", "SN_theta_2"])

    def get_log_likelihood(self, data):
        luminosity = data["double_luminosity"]
        theta1 = data["SN_theta_1"]
        theta2 = data["SN_theta_2"]
        if theta2 < 0:
            return -np.inf
        return -np.sum((luminosity - theta1) * (luminosity - theta1) / (theta2 * theta2)) - luminosity.size * np.log(np.sqrt(2 * np.pi) * theta2)


class ExampleModel(Model):
    r"""An implementation of :class:`.ExampleLatent` using classes instead of procedural code.
    """

    def __init__(self):
        super(ExampleModel, self).__init__()

        n = 200

        flux = ObservedFlux(n=n)
        luminosity = LatentLuminosity(n=n)
        useless = UselessTransformation()
        supernova = UnderlyingSupernovaDistribution()
        self.add_node(flux)
        self.add_node(luminosity)
        self.add_node(useless)
        self.add_node(supernova)

        self.add_edge(FluxToLuminosity())
        self.add_edge(LuminosityToAdjusted())
        self.add_edge(LuminosityToSupernovaDistribution())

        self.finalise()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    exampleModel = ExampleModel()
    exampleModel.fit_model(filename="exampleLatentClass")
