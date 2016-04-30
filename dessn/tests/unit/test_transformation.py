from ...framework.model import Model
from ...framework.parameter import ParameterUnderlying, ParameterObserved, ParameterTransformation
from ...framework.edge import EdgeTransformation, Edge
import numpy as np


class ObservedValue(ParameterObserved):
    def __init__(self, data):
        super().__init__("obs", "$o$", data)


class TransformedValue(ParameterTransformation):
    def __init__(self, n):
        super().__init__("transformed", "$t$")
        self.n = n


class Underlying(ParameterUnderlying):
    def __init__(self):
        super().__init__("under", "$u$")

    def get_suggestion(self, data):
        return np.mean(data["obs"])

    def get_suggestion_sigma(self, data):
        return np.sqrt(len(data["obs"])) * np.std(data["obs"])

    def get_suggestion_requirements(self):
        return ["obs"]

    def get_log_prior(self, data):
        return 1


class ToTransformed(EdgeTransformation):
    def __init__(self):
        super().__init__("transformed", "obs")

    def get_transformation(self, data):
        return {"transformed": 100 * data["obs"]}


class ToUnderlying(Edge):
    def __init__(self):
        super().__init__("transformed", "under")
        self.prefactor = np.log(np.sqrt(2 * np.pi) * 1)

    def get_log_likelihood(self, data):
        diff = data["transformed"] - data["under"]
        return -(diff * diff) / (2.0) - self.prefactor


class TransformedModel(Model):
    def __init__(self):
        super().__init__("TransformedModel")
        raw_data = np.random.normal(loc=50, scale=1, size=200)
        data = raw_data / 100
        self.raw_data = raw_data
        self.raw_data2 = data
        self.add_node(ObservedValue(data))
        self.add_node(TransformedValue(len(data)))
        self.add_node(Underlying())
        self.add_edge(ToTransformed())
        self.add_edge(ToUnderlying())
        self.finalise()


class TestTransformed(object):
    model = TransformedModel()
    theta = [50]

    def test_num_parameters(self):
        assert len(self.model._theta_names) == 1

    def test_latent_prior(self):
        assert self.model.get_log_prior(self.theta) == 1.0

    def test_latent_posterior(self):
        posterior = np.sum(-(0.5 * (self.theta[0] - self.model.raw_data) ** 2)
                           - np.log(np.sqrt(2 * np.pi) * 1))
        posterior += 1
        model_posterior = self.model.get_log_posterior(self.theta)
        assert np.isclose(model_posterior, posterior)
