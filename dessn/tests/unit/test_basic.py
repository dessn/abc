from ...framework.model import Model
from ...framework.parameter import ParameterUnderlying, ParameterObserved
from ...framework.edge import Edge
import numpy as np


class CoinFlipRate(ParameterUnderlying):
    def __init__(self):
        super().__init__("r", "$r$")

    def get_suggestion_sigma(self, data):
        return 0.4

    def get_suggestion(self, data):
        return 0.5

    def get_log_prior(self, data):
        return np.log(0.5) if data["r"] < 0.9 else np.log(0.001)


class ObservedCoinTosses(ParameterObserved):
    def __init__(self):
        super().__init__("f_o", r"$\hat{f}$", np.array([1, 1, 1, 0]))


class ToRate(Edge):
    def __init__(self):
        super().__init__("f_o", "r")

    def get_log_likelihood(self, data):
        f_o = data["f_o"]
        r = data["r"]
        return np.log(f_o * r + (1 - f_o) * (1 - r))


class CoinModel(Model):
    def __init__(self):
        super(CoinModel, self).__init__("CoinToss")
        self.add_node(ObservedCoinTosses())
        self.add_node(CoinFlipRate())
        self.add_edge(ToRate())
        self.finalise()


def test_model_add_node():
    model = Model("test")
    node = ObservedCoinTosses()
    model.add_node(node)
    assert len(model.nodes) == 1
    assert model.nodes[0] == node


class TestClass(object):
    model = CoinModel()

    def test_basic_prior1(self):
        assert np.isclose(self.model.get_log_prior([0.5]), np.log(0.5))

    def test_basic_prior2(self):
        assert np.isclose(self.model.get_log_prior([0.99]), np.log(0.001))

    def test_basic_posterior1(self):
        theta = [0.5]
        assert np.isclose(self.model.get_log_posterior(theta),
                          np.log(0.5) + np.log(0.5 ** 4))

    def test_basic_posterior2(self):
        theta = [0.9]
        assert np.isclose(self.model.get_log_posterior(theta),
                          np.log(0.001) + np.sum(np.log([0.9, 0.9, 0.9, 0.1])))

    def test_basic_likelihood1(self):
        theta = [0.5]
        assert np.isclose(self.model.get_log_likelihood(theta),
                          np.log(0.5 ** 4))

    def test_basic_likelihood2(self):
        theta = [0.9]
        assert np.isclose(self.model.get_log_likelihood(theta),
                          np.sum(np.log([0.9, 0.9, 0.9, 0.1])))

    def test_num_parameters(self):
        assert len(self.model._theta_names) == 1
