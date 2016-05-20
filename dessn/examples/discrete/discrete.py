import logging
import os
import sys
import numpy as np
from dessn.framework.edge import Edge
from dessn.framework.model import Model
from dessn.framework.parameter import ParameterObserved, ParameterUnderlying, ParameterDiscrete
from dessn.framework.samplers.ensemble import EnsembleSampler


def get_data(n=200):
    np.random.seed(0)
    rate = 0.7
    colour = np.random.random(n) <= rate
    size = 1 + 1.0 * colour + np.random.normal(scale=0.3, size=n)

    misidentification = np.random.random(n) > 0.9
    final = (colour ^ misidentification)
    colours = ['red' if a else 'blue' for a in final]
    return size.tolist(), colours


class ObservedColour(ParameterObserved):
    def __init__(self):
        sizes, colours = get_data()
        super(ObservedColour, self).__init__("c_o", "$c_o$", colours, group="Obs. Colour")


class ObservedSize(ParameterObserved):
    def __init__(self):
        sizes, colours = get_data()
        super(ObservedSize, self).__init__("s_o", "$s_o$", sizes, group="Obs. Size")


class Colour(ParameterDiscrete):
    def __init__(self):
        super(Colour, self).__init__("c", "$c$", group="Colour")

    def get_discrete(self, data):
        return "red", "blue"

    def get_discrete_requirements(self):
        return []


class UnderlyingRate(ParameterUnderlying):
    def __init__(self):
        super(UnderlyingRate, self).__init__("r", "$r$", group="Rate")

    def get_log_prior(self, data):
        r = data["r"]
        if r < 0 or r > 1:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return 0.5

    def get_suggestion_sigma(self, data):
        return 0.05

    def get_suggestion_requirements(self):
        return []


class ToColour(Edge):

    def __init__(self):
        super(ToColour, self).__init__("c_o", "c")

    def get_log_likelihood(self, data):
        c = data["c"]
        c_o = data["c_o"]
        m = 0.9
        prob = m * (c == c_o) + (1 - m) * (c != c_o)
        return np.log(prob)


class ToColour2(Edge):
    def __init__(self):
        super(ToColour2, self).__init__("s_o", "c")
        self.sqrt2pi = np.sqrt(2 * np.pi)

    def get_log_likelihood(self, data):
        c = data["c"]
        s_o = data["s_o"]
        mid = 1 + 1.0 * (c == "red")
        sigma = 0.3
        ps = -(s_o - mid) * (s_o - mid) / (2 * sigma * sigma) - np.log(self.sqrt2pi * sigma)
        return ps


class ToRate(Edge):
    def __init__(self):
        super(ToRate, self).__init__("c", "r")

    def get_log_likelihood(self, data):
        c = data["c"]
        r = data["r"]
        probs = r * (c == "red") + (1 - r) * (c != "red")
        return np.log(probs)


class DiscreteModel(Model):
    r"""A small example framework illustrating how to use discrete parameters.

    As normal, the framework is set up by declaring parameters (which can be thought of like nodes on a PGM),
    and declaring the edges between parameters (the conditional probabilities).

    This is the primary class in this package, and you can see that other classes
    inherit from either :class:`.Parameter` or from :class:`.Edge`.

    I leave the documentation for :class:`.Parameter` and :class:`.Edge` to those classes,
    and encourage viewing the code directly to understand exactly what is happening.

    Running this file in python first generates a PGM of the framework, and then runs ``emcee`` and creates a corner plot:

    .. figure::     ../dessn/examples/discrete/output/pgm.png
        :align:     center

    .. figure::     ../dessn/examples/discrete/output/surfaces.png
        :align:     center

    """

    def __init__(self):
        super(DiscreteModel, self).__init__("Discrete")

        self.add_node(ObservedColour())
        self.add_node(ObservedSize())
        self.add_node(Colour())
        self.add_node(UnderlyingRate())
        self.add_edge(ToColour())
        self.add_edge(ToColour2())
        self.add_edge(ToRate())

        self.finalise()

if __name__ == "__main__":
    model = DiscreteModel()
    only_data = len(sys.argv) > 1
    if only_data:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    else:
        logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(__file__)
    temp_dir = os.path.abspath(dir_name + "/output/data")
    plot_file = os.path.abspath(dir_name + "/output/surfaces.png")

    if not only_data:
        pgm_file = os.path.abspath(dir_name + "/output/pgm.png")
        model.get_pgm(pgm_file)

    logging.info("Starting fit")
    sampler = EnsembleSampler(num_steps=3000, num_burn=500, temp_dir=temp_dir, save_interval=20)
    chain_consumer = model.fit(sampler)

    if not only_data:
        print(chain_consumer.get_summary())
        chain_consumer.plot(filename=plot_file, truth=[0.7])
