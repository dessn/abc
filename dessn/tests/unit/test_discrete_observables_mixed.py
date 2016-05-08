from ...framework.model import Model
from ...framework.parameter import ParameterUnderlying, ParameterObserved, ParameterDiscrete
from ...framework.edge import Edge
import numpy as np


class ObservedValue(ParameterObserved):
    def __init__(self, data):
        super().__init__("obs", "$o$", data)


class DiscreteValue(ParameterDiscrete):
    def __init__(self):
        super().__init__("discrete", "$l$")

    def get_discrete(self, data):
        return [list(a.keys()) if type(a) == dict else a for a in data["obs"]]

    def get_discrete_requirements(self):
        return ["obs"]


class Underlying(ParameterUnderlying):
    def __init__(self):
        super().__init__("under", "$u$")

    def get_suggestion(self, data):
        return 0.5

    def get_suggestion_sigma(self, data):
        return 0.3

    def get_log_prior(self, data):
        return 1


class ToDiscrete(Edge):
    def __init__(self):
        super().__init__("obs", "discrete")

    def get_log_likelihood(self, data):
        probs = []
        for obs, d in zip(data["obs"], data["discrete"]):
            if type(obs) == dict:
                probs.append(obs[d])
            else:
                if d == "heads":
                    probs.append(1)
                else:
                    probs.append(0)
        probs = np.array(probs)
        return np.log(probs)


class ToUnderlying(Edge):
    def __init__(self):
        super().__init__("discrete", "under")

    def get_log_likelihood(self, data):
        mask = np.array([a == "heads" for a in data["discrete"]])
        r = data["under"]
        result = np.log(mask * r + (1 - mask) * (1 - r))
        return result


class DiscreteModelTest4(Model):
    def __init__(self):
        super().__init__("DiscreteModel")
        data = [{"heads": 0.8, "tails": 0.2}, "heads"]
        self.raw_data = data
        self.add_node(ObservedValue(data))
        self.add_node(DiscreteValue())
        self.add_node(Underlying())
        self.add_edge(ToDiscrete())
        self.add_edge(ToUnderlying())
        self.finalise()


class TestDiscreteMixed(object):
    model = DiscreteModelTest4()
    theta = [0.6]

    def test_latent_num_parameters(self):
        assert len(self.model._theta_names) == 1

    def test_latent_prior(self):
        assert self.model.get_log_prior(self.theta) == 1.0

    def test_latent_posterior(self):
        posterior = np.log((0.6 * 0.8 + 0.4 * 0.2) * (0.6 * 1 + 0.4 * 0))
        posterior += 1
        model_posterior = self.model.get_log_posterior(self.theta)
        assert np.isclose(model_posterior, posterior)
