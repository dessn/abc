from ...framework.model import Model
from ...framework.parameter import ParameterUnderlying, ParameterObserved
from ...framework.edge import Edge
import numpy as np
import pytest


class PriorNan(ParameterUnderlying):
    def __init__(self):
        super().__init__("prior", "prior")

    def get_suggestion(self, data):
        return 1.0

    def get_suggestion_sigma(self, data):
        return 1.0

    def get_log_prior(self, data):
        return np.nan


class PriorNormal(ParameterUnderlying):
    def __init__(self):
        super().__init__("prior", "prior")

    def get_suggestion(self, data):
        return 1.0

    def get_suggestion_sigma(self, data):
        return 1.0

    def get_log_prior(self, data):
        return 1.0


class Observed(ParameterObserved):
    def __init__(self):
        super().__init__("obs", "obs", np.array([0.0]))


class EdgeNan(Edge):
    def __init__(self):
        super().__init__("obs", "prior")

    def get_log_likelihood(self, data):
        return np.nan


class EdgeNormal(Edge):
    def __init__(self):
        super().__init__("obs", "prior")

    def get_log_likelihood(self, data):
        return 1.0


def test_nan_prior():
    m = Model("model")
    m.add_node(Observed())
    m.add_node(PriorNan())
    m.add_edge(EdgeNormal())
    m.finalise()
    with pytest.raises(ValueError) as e:
        m.get_log_posterior([0])
    assert "nan" in str(e.value).lower()


def test_nan_edge():
    m = Model("model")
    m.add_node(Observed())
    m.add_node(PriorNormal())
    m.add_edge(EdgeNan())
    m.finalise()
    with pytest.raises(ValueError) as e:
        m.get_log_posterior([0])
    assert "nan" in str(e.value).lower()
