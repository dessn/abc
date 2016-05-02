from ...framework.model import Model
from ...framework.parameter import ParameterUnderlying, ParameterObserved, ParameterLatent, \
    ParameterTransformation
from ...framework.edge import Edge
import numpy as np
import pytest


def test_fail_without_underlying():
    m = Model("name")
    n1 = ParameterObserved("a", "a", np.random.random(20))
    m.add_node(n1)
    with pytest.raises(AssertionError) as e:
        m.finalise()
    assert "underlying" in str(e.value).lower()


def test_fail_without_observed():
    m = Model("name")
    m.add_node(ParameterUnderlying("a", "a"))
    with pytest.raises(AssertionError) as e:
        m.finalise()
    assert "observed" in str(e.value).lower()


def test_fail_without_edges():
    m = Model("name")
    m.add_node(ParameterObserved("a", "a", np.random.random(20)))
    m.add_node(ParameterUnderlying("b", "b"))
    with pytest.raises(AssertionError) as e:
        m.finalise()
    assert "unconnected" in str(e.value).lower()


def test_fail_on_mismatched_data():
    m = Model("name")
    n1 = ParameterObserved("a", "a", np.random.random(20))
    n2 = ParameterObserved("b", "b", np.random.random(10))
    n3 = ParameterUnderlying("c", "c")
    e1 = Edge("a", "c")
    e2 = Edge("b", "c")
    m.add_node(n1)
    m.add_node(n2)
    m.add_node(n3)
    m.add_edge(e1)
    m.add_edge(e2)
    with pytest.raises(AssertionError) as e:
        m.finalise()
    assert "data size" in str(e.value).lower()


def test_fail_on_duplicate_names():
    m = Model("name")
    n1 = ParameterObserved("a", "a", np.random.random(20))
    n2 = ParameterObserved("a", "a", np.random.random(10))
    m.add_node(n1)
    with pytest.raises(AssertionError) as e:
        m.add_node(n2)
    assert "already in the framework" in str(e.value).lower()


def test_unorderable_edges():
    m = Model("name")
    m.add_node(ParameterObserved("a", "a", np.random.random(20)))
    m.add_node(ParameterTransformation("b", "b"))
    m.add_node(ParameterTransformation("d", "d"))
    m.add_node(ParameterUnderlying("c", "c"))
    m.add_edge(Edge("a", "b"))
    m.add_edge(Edge("b", "d"))
    m.add_edge(Edge("d", "b"))
    m.add_edge(Edge("b", "c"))
    with pytest.raises(AssertionError) as e:
        m.finalise()
    assert "cannot be ordered" in str(e.value).lower()


class PriorNan(ParameterUnderlying):
    def get_suggestion(self, data):
        pass

    def get_log_prior(self, data):
        def __init__(self):
            super().__init__("a", "a")

    # def test_fail_without_edges():
#     pass
