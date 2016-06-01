from dessn.framework.edge import Edge
from dessn.framework.model import Model
from dessn.framework.parameter import ParameterObserved, ParameterUnderlying
from dessn.framework.samplers.ensemble import EnsembleSampler
import numpy as np
import os


class Observed(ParameterObserved):
    def __init__(self):
        super().__init__("data", "$d$", np.random.normal(size=1000))


class Mean(ParameterUnderlying):
    def __init__(self):
        super().__init__("mean", r"$\mu$")

    def get_log_prior(self, data):
        return 1

    def get_suggestion(self, data):
        return 0

    def get_suggestion_sigma(self, data):
        return 3


class ConditionalProbability(Edge):
    def __init__(self):
        super().__init__("data", "mean")

    def get_log_likelihood(self, data):
        return -(data["data"] - data["mean"])**2


class Example(Model):
    def __init__(self):
        super().__init__("Example model")
        self.add_node(Observed())
        self.add_node(Mean())
        self.add_edge(ConditionalProbability())

if __name__ == "__main__":
    directory = os.path.dirname(__file__) + os.sep + "output/"
    model = Example()
    model.get_pgm(filename=directory + "pgm.png")
    sampler = EnsembleSampler(temp_dir=directory)
    c = model.fit(sampler)
    c.plot(filename=directory + "surface.png", truth=[0])
