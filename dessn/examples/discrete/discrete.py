import logging
import os
import sys

import numpy as np

from dessn.model.edge import Edge
from dessn.model.model import Model
from dessn.model.node import NodeObserved, NodeUnderlying, NodeDiscrete


def get_data(n=50):
    np.random.seed(0)
    colour = np.random.random(n) > 0.5
    size = 1 + 1.0 * colour + np.random.normal(scale=0.3, size=n)

    misidentification = np.random.random(n) > 0.9
    final = (colour ^ misidentification)
    colours = ['red' if a else 'blue' for a in final]
    return size.tolist(), colours


class ObservedColour(NodeObserved):
    def __init__(self):
        sizes, colours = get_data()
        super(ObservedColour, self).__init__("c_o", "$c_o$", colours, group="Obs. Colour")


class ObservedSize(NodeObserved):
    def __init__(self):
        sizes, colours = get_data()
        super(ObservedSize, self).__init__("s_o", "$s_o$", sizes, group="Obs. Size")


class Colour(NodeDiscrete):
    def __init__(self):
        super(Colour, self).__init__("c", "$c$", group="Colour")

    def get_discrete(self, data):
        return ["red", "blue"]

    def get_discrete_requirements(self):
        return []


class UnderlyingRate(NodeUnderlying):
    def __init__(self):
        super(UnderlyingRate, self).__init__("r", "$r$", group="Rate")

    def get_log_prior(self, data):
        r = data["r"]
        if r < 0 or r > 1:
            return -np.inf
        return 1

    def get_suggestion(self, data):
        return [0.5]

    def get_suggestion_requirements(self):
        return []


class ToColour(Edge):

    def __init__(self):
        super(ToColour, self).__init__("c_o", "c")

    def get_log_likelihood(self, data):
        c = data["c"]
        c_o = data["c_o"]
        return np.log(0.9 if c == c_o else 0.1)


class ToColour2(Edge):
    def __init__(self):
        super(ToColour2, self).__init__("s_o", "c")

    def get_log_likelihood(self, data):
        c = data["c"]
        s_o = data["s_o"]
        if c == "red":
            mid = 2
        else:
            mid = 1
        sigma = 0.3
        ps = -(s_o - mid) * (s_o - mid) / (2 * sigma * sigma) - np.log(np.sqrt(2 * np.pi) * sigma)
        return ps


class ToRate(Edge):
    def __init__(self):
        super(ToRate, self).__init__("c", "r")

    def get_log_likelihood(self, data):
        c = data["c"]
        r = data["r"]
        if c == "red":
            return np.log(r)
        else:
            return np.log(1 - r)


class DiscreteModel(Model):
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
    temp_dir = os.path.abspath(dir_name + "/../../../temp/discrete")

    if not only_data:
        plot_file = os.path.abspath(dir_name + "/../../../plots/discrete.png")
        pgm_file = os.path.abspath(dir_name + "/../../../plots/discretePGM.png")
        # model.get_pgm(pgm_file)

    logging.info("Starting fit")

    for x in np.linspace(0.1, 0.9, 21):
        pass
        print(x, model._get_log_posterior([x]))

    '''
    model.fit_model(num_steps=500, num_burn=100, temp_dir=temp_dir, save_interval=20)

    if not only_data:
        chain_consumer = model.get_consumer()
        chain_consumer.configure_general(bins=1.0)
        print(chain_consumer.get_summary())
        chain_consumer.plot(filename=plot_file, display=False, truth=[0.3])'''