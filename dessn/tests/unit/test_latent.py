from ...framework.model import Model
from ...framework.parameter import ParameterUnderlying, ParameterObserved, ParameterLatent
from ...framework.edge import Edge
import numpy as np


class ObservedValue(ParameterObserved):
    def __init__(self, data):
        super().__init__("obs", "$o$", data)


class LatentValue(ParameterLatent):
    def __init__(self, n):
        super().__init__("latent", "$l$")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return ["obs"]

    def get_suggestion(self, data):
        return data["obs"]

    def get_suggestion_sigma(self, data):
        return 2 * np.std(data["obs"]) * np.ones(data["obs"].shape)


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


class ToLatent(Edge):
    def __init__(self):
        super().__init__("obs", "latent")
        self.prefactor = np.log(np.sqrt(2 * np.pi) * 0.3)

    def get_log_likelihood(self, data):
        diff = data["obs"] - data["latent"]
        return -(diff * diff) / (2 * 0.3 * 0.3) - self.prefactor


class ToUnderlying(Edge):
    def __init__(self):
        super().__init__("latent", "under")
        self.prefactor = np.log(np.sqrt(2 * np.pi) * 1)

    def get_log_likelihood(self, data):
        diff = data["latent"] - data["under"]
        return -(diff * diff) / (2.0) - self.prefactor


class LatentModel(Model):
    def __init__(self):
        super().__init__("LatentModel")
        data = np.random.normal(loc=5, scale=1, size=200)
        error_data = data + np.random.normal(loc=0, scale=0.3, size=200)
        self.raw_data = data
        self.raw_error_data = error_data
        self.add_node(ObservedValue(error_data))
        self.add_node(LatentValue(len(data)))
        self.add_node(Underlying())
        self.add_edge(ToLatent())
        self.add_edge(ToUnderlying())
        self.finalise()


class TestLatent(object):
    model = LatentModel()
    theta = [5] + model.raw_data.tolist()

    def test_latent_num_parameters(self):
        assert len(self.model._theta_names) == 1 + len(self.model.raw_data)

    def test_latent_prior(self):
        assert self.model.get_log_prior(self.theta) == 1.0

    def test_latent_posterior(self):
        posterior = np.sum(-(0.5 * (self.theta[0] - np.array(self.theta[1:])) ** 2)
                           - np.log(np.sqrt(2 * np.pi) * 1))
        posterior += np.sum(-((np.array(self.theta[1:]) - self.model.raw_error_data) ** 2)
                            / (2 * 0.3 ** 2)
                            - np.log(np.sqrt(2 * np.pi) * 0.3))
        posterior += 1
        model_posterior = self.model.get_log_posterior(self.theta)
        assert np.isclose(model_posterior, posterior)

    def test_suggestion(self):
        suggestion = self.model._get_suggestion()
        obs = self.model.raw_error_data
        expected = [np.mean(obs)] + self.model.raw_error_data.tolist()
        assert suggestion == expected

    def test_suggestion_sigma(self):
        obs = self.model.raw_error_data
        sigma = self.model._get_suggestion_sigma()
        sqrt = np.sqrt(len(obs)) * np.std(obs)
        expected = [sqrt] + [2 * np.std(obs)] * len(obs)
        assert sigma == expected
