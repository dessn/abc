import logging
import os
import sys

import numpy as np

from dessn.examples.simple.example import Example
from dessn.framework.edge import Edge, EdgeTransformation
from dessn.framework.model import Model
from dessn.framework.parameter import ParameterObserved, ParameterLatent, ParameterUnderlying, ParameterTransformation


class ObservedFlux(ParameterObserved):
    def __init__(self, n=100):
        self.n = n
        flux, error = Example.get_data(n=n, scale=0.5)
        super(ObservedFlux, self).__init__("flux", "$f$", flux, group="Flux")


class ObservedFluxError(ParameterObserved):
    def __init__(self, n=100):
        self.n = n
        flux, error = Example.get_data(n=n, scale=0.5)
        super(ObservedFluxError, self).__init__("flux_error", r"$\sigma_f$", error, group="Flux")


class LatentLuminosity(ParameterLatent):

    def __init__(self, n=100):
        super(LatentLuminosity, self).__init__("luminosity", "$L$", group="Luminosity")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return ["flux"]

    def get_suggestion(self, data):
        return data["flux"]

    def get_suggestion_sigma(self, data):
        return data["flux"] * 0.05


class UnderlyingSupernovaDistribution1(ParameterUnderlying):

    def get_log_prior(self, data):
        """ We framework the prior enforcing realistic values"""
        mean = data["SN_theta_1"]
        if mean < 0 or mean > 200:
            return -np.inf
        return 1

    def __init__(self):
        super(UnderlyingSupernovaDistribution1, self).__init__("SN_theta_1", r"$\theta_1$", group="Supernova")

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 50

    def get_suggestion_sigma(self, data):
        return 5


class UnderlyingSupernovaDistribution2(ParameterUnderlying):

    def get_log_prior(self, data):
        """ We framework the prior enforcing realistic values"""
        sigma = data["SN_theta_2"]
        if sigma < 0 or sigma > 100:
            return -np.inf
        return 1

    def __init__(self):
        super(UnderlyingSupernovaDistribution2, self).__init__("SN_theta_2", r"$\theta_2$", group="Supernova")

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 5 # Deliberately wrong

    def get_suggestion_sigma(self, data):
        return 0.5


class UselessTransformation(ParameterTransformation):
    def __init__(self):
        super(UselessTransformation, self).__init__("double_luminosity", "$2L$", group="Transformed Luminosity")


class LuminosityToAdjusted(EdgeTransformation):
    def __init__(self):
        super(LuminosityToAdjusted, self).__init__("double_luminosity", "luminosity")

    def get_transformation(self, data):
        return {"double_luminosity": data["luminosity"] * 2.0}


class FluxToLuminosity(Edge):
    def __init__(self):
        super(FluxToLuminosity, self).__init__(["flux", "flux_error"], "luminosity")

    def get_log_likelihood(self, data):
        l = data["luminosity"]
        f = data["flux"]
        e = data["flux_error"]
        return -((f - l) * (f - l) / (2.0 * e * e) - np.log(np.sqrt(2 * np.pi) * e))


class LuminosityToSupernovaDistribution(Edge):
    def __init__(self):
        super(LuminosityToSupernovaDistribution, self).__init__("double_luminosity", ["SN_theta_1", "SN_theta_2"])

    def get_log_likelihood(self, data):
        l = data["double_luminosity"]
        t1 = data["SN_theta_1"]
        t2 = data["SN_theta_2"]
        if t2 < 0:
            return -np.inf
        return -(l - t1) * (l - t1) / (2.0 * t2 * t2) - np.log(np.sqrt(2 * np.pi) * t2)


class ExampleModel(Model):
    r"""An implementation of :class:`.ExampleLatent` using classes instead of procedural code.

    The framework is set up by declaring nodes, the edges between nodes, and then calling ``finalise`` on the framework
    to verify its correctness.

    This is the primary class in this package, and you can see that other classes inherit from either
    :class:`.Parameter` or from :class:`.Edge`.

    I leave the documentation for :class:`.Parameter` and :class:`.Edge` to those classes, and encourage viewing the
    code directly to understand exactly what is happening.

    Running this file in python first generates a PGM of the framework, and then runs ``emcee`` and creates a corner plot:

    .. figure::     ../plots/exampleModel.png
        :align:     center

    .. figure::     ../plots/examplePGM.png
        :align:     center

    .. figure::     ../plots/exampleModelWalk.png
        :align:     center

    We could also run the example framework using the PT sampler by specifying a number of temperature to the ``fit_model``
    method. You would get similar results.
    """

    def __init__(self):
        super(ExampleModel, self).__init__("ExampleModel")

        n = 30

        flux = ObservedFlux(n=n)
        flux_error = ObservedFluxError(n=n)
        luminosity = LatentLuminosity(n=n)
        useless = UselessTransformation()
        supernova1 = UnderlyingSupernovaDistribution1()
        supernova2 = UnderlyingSupernovaDistribution2()
        self.add_node(flux)
        self.add_node(flux_error)
        self.add_node(luminosity)
        self.add_node(useless)
        self.add_node(supernova1)
        self.add_node(supernova2)

        self.add_edge(FluxToLuminosity())
        self.add_edge(LuminosityToAdjusted())
        self.add_edge(LuminosityToSupernovaDistribution())

        self.finalise()


if __name__ == "__main__":
    only_data = len(sys.argv) > 1
    if only_data:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    else:
        logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(__file__)
    logging.info("Creating framework")
    exampleModel = ExampleModel()
    temp_dir = os.path.abspath(dir_name + "/../../../../temp/exampleModel")

    if not only_data:
        plot_file = os.path.abspath(dir_name + "/../../../../plots/exampleModel.png")
        plot_file2 = os.path.abspath(dir_name + "/../../../../plots/exampleModelWalk.png")
        pgm_file = os.path.abspath(dir_name + "/../../../../plots/examplePGM.png")
        # exampleModel.get_pgm(pgm_file)

    logging.info("Starting fit")
    exampleModel.fit_model(num_steps=15000, num_burn=5000, temp_dir=temp_dir, save_interval=20)

    if not only_data:
        chain_consumer = exampleModel.get_consumer()
        chain_consumer.configure_general(bins=0.8)
        print(chain_consumer.get_summary())
        chain_consumer.plot_walks(filename=plot_file2)
        chain_consumer.plot(filename=plot_file, display=False, figsize="PAGE", truth=[100, 20])
