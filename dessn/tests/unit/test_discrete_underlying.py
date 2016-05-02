from ...framework.model import Model
from ...framework.parameter import ParameterUnderlying, ParameterObserved, ParameterDiscrete
from ...framework.edge import Edge
import numpy as np
import pytest


class ObservedValue(ParameterObserved):
    def __init__(self, data):
        super().__init__("obs", "$o$", data)


class TypeMean(ParameterDiscrete):
    def __init__(self):
        super().__init__("discrete", "$l$")

    def get_discrete(self, data):
        return 0, 3


class TypeMeanFailure(ParameterDiscrete):
    def __init__(self):
        super().__init__("discrete", "$l$")

    def get_discrete(self, data):
        return "oops"


class Rate(ParameterUnderlying):
    def __init__(self):
        super().__init__("under", "$u$")

    def get_suggestion(self, data):
        return 0.5

    def get_suggestion_sigma(self, data):
        return 0.4

    def get_log_prior(self, data):
        return 1


class ToTypeMean(Edge):
    def __init__(self):
        super().__init__("obs", "discrete")

    def get_log_likelihood(self, data):
        obs = data["obs"]
        mean = data["discrete"]
        diff = obs - mean
        return -0.5 * diff * diff - np.log(np.sqrt(2 * np.pi))


class ToUnderlying(Edge):
    def __init__(self):
        super().__init__("discrete", "under")

    def get_log_likelihood(self, data):
        mask = np.array(data["discrete"]) == 3
        print(data["discrete"])

        r = data["under"]
        result = np.log(mask * r + (1 - mask) * (1 - r))
        return result


class DiscreteModel(Model):
    def __init__(self):
        super().__init__("DiscreteModel")
        rate = 0.7
        r1 = np.random.normal(loc=0, scale=1, size=100)
        r2 = np.random.normal(loc=3, scale=1, size=100)
        m = np.random.random(size=100) > rate
        data = r1 * m + (1 - m) * r2
        self.raw_data = data
        self.add_node(ObservedValue(data))
        self.add_node(TypeMean())
        self.add_node(Rate())
        self.add_edge(ToTypeMean())
        self.add_edge(ToUnderlying())
        self.finalise()


class DiscreteModelFailure(Model):
    def __init__(self):
        super().__init__("DiscreteModel")
        rate = 0.7
        r1 = np.random.normal(loc=0, scale=1, size=100)
        r2 = np.random.normal(loc=3, scale=1, size=100)
        m = np.random.random(size=100) > rate
        data = r1 * m + (1 - m) * r2
        self.raw_data = data
        self.add_node(ObservedValue(data))
        self.add_node(TypeMeanFailure())
        self.add_node(Rate())
        self.add_edge(ToTypeMean())
        self.add_edge(ToUnderlying())
        self.finalise()


class TestDiscrete(object):
    model = DiscreteModel()
    theta = [0.7]

    def test_latent_num_parameters(self):
        assert len(self.model._theta_names) == 1

    def test_latent_prior(self):
        assert self.model.get_log_prior(self.theta) == 1.0

    def test_latent_posterior(self):
        posterior = 1
        rate = self.theta[0]
        for d in self.model.raw_data:
            prob1 = 1/(np.sqrt(2 * np.pi)) * np.exp(-0.5 * (d - 0)**2) * (1 - rate)
            prob2 = 1 / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * (d - 3) ** 2) * rate
            prob = prob1 + prob2
            posterior += np.log(prob)
        model_posterior = self.model.get_log_posterior(self.theta)
        assert np.isclose(model_posterior, posterior)

    def test_suggestion(self):
        suggestion = self.model._get_suggestion()
        assert suggestion == [0.5]

    def test_suggestion_sigma(self):
        sigma = self.model._get_suggestion_sigma()
        assert sigma == [0.4]


def test_discrete_failure():
    model_failure = DiscreteModelFailure()
    with pytest.raises(ValueError) as e:
        model_failure.get_log_posterior([0.7])
    assert "not a tuple or a list" in str(e.value).lower()
