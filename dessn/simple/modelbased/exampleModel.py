import numpy as np
from dessn.model.model import Model
from dessn.model.node import NodeObserved, NodeLatent, NodeUnderlying, NodeTransformation
from dessn.model.edge import Edge, EdgeTransformation
from dessn.simple.example import Example
import logging
import os


class ObservedFlux(NodeObserved):
    def __init__(self, n=100):
        self.n = n
        flux, error = Example.get_data(n=n, scale=0.5)

        super(ObservedFlux, self).__init__("Flux", ["flux", "flux_error"], ["$f$", r"$\sigma_f$"], [flux, error])


class LatentLuminosity(NodeLatent):
    def __init__(self, n=100):
        super(LatentLuminosity, self).__init__("Luminosity", "luminosity", "$L$")
        self.n = n

    def get_num_latent(self):
        return self.n


class UnderlyingSupernovaDistribution(NodeUnderlying):
    def get_log_prior(self, data):
        return 1

    def __init__(self):
        super(UnderlyingSupernovaDistribution, self).__init__("Supernova", ["SN_theta_1", "SN_theta_2"],
                                                              [r"$\theta_1$", r"$\theta_2$"])


class UselessTransformation(NodeTransformation):
    def __init__(self):
        super(UselessTransformation, self).__init__("Transformed Luminosity", "double_luminosity", "$2L$")


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
        return -np.sum((flux - luminosity) * (flux - luminosity) / (2.0 * flux_error * flux_error) + np.log(np.sqrt(2 * np.pi) * flux_error))


class LuminosityToSupernovaDistribution(Edge):
    def __init__(self):
        super(LuminosityToSupernovaDistribution, self).__init__("double_luminosity", ["SN_theta_1", "SN_theta_2"])

    def get_log_likelihood(self, data):
        luminosity = data["double_luminosity"]
        theta1 = data["SN_theta_1"]
        theta2 = data["SN_theta_2"]
        if theta2 < 0:
            return -np.inf
        return -np.sum((luminosity - theta1) * (luminosity - theta1) / (2.0 * theta2 * theta2)) - luminosity.size * np.log(np.sqrt(2 * np.pi) * theta2)


class ExampleModel(Model):
    r"""An implementation of :class:`.ExampleLatent` using classes instead of procedural code.

    The model is set up by declaring nodes, the edges between nodes, and then calling ``finalise`` on the model
    to verify its correctness.

    This is the primary class in this package, and you can see that other classes inherit from either :class:`.Node` or from :class:`.Edge`.

    I leave the documentation for :class:`.Node`s and :class:`.Edge`s to those classes, and encourage viewing the code directly
    to understand exactly what is happening.

    Running this file in python first generates a PGM of the model, and then runs ``emcee`` and creates a corner plot:

    .. figure::     ../plots/exampleModel.png
        :align:     center

    .. figure::     ../plots/examplePGM.png
        :align:     center
    """

    def __init__(self):
        super(ExampleModel, self).__init__("ExampleModel")

        n = 30

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
    dir_name = os.path.dirname(__file__)
    exampleModel = ExampleModel()
    corner_file = os.path.abspath(dir_name + "/../../../plots/exampleModel.png")
    temp_dir = os.path.abspath(dir_name + "/../../../temp/exampleModel")
    pgm_file = os.path.abspath(dir_name + "/../../../plots/examplePGM.png")
    exampleModel.get_pgm(pgm_file)
    exampleModel.fit_model(num_steps=1000, num_burn=500, temp_dir=temp_dir, save_interval=5)
    exampleModel.corner(corner_file)
    exampleModel.chain_plot()
    exampleModel.chain_summary()
