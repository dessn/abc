from ...framework.model import Model
from ...framework.parameter import ParameterUnderlying, ParameterObserved, ParameterLatent
from ...framework.edge import Edge
import numpy as np
import pytest


def test_fail_without_underlying():
    m = Model("name")
    n1 = ParameterObserved("a", "a", np.random.random(20))
    m.add_node(n1)
    with pytest.raises(AssertionError) as e:
        m.finalise()
    assert "underlying" in str(e.value)


def test_fail_without_observed():
    m = Model("name")
    m.add_node(ParameterUnderlying("a", "a"))
    with pytest.raises(AssertionError) as e:
        m.finalise()
    assert "observed" in str(e.value)


# def test_fail_without_edges():
#     pass

        # def test_fail_on_mismatched_data():
#     m = Model("name")
#     n1 = ParameterObserved("a", "a", np.random.random(20))
#     n2 = ParameterObserved("b", "b", np.random.random(10))
#     m.add_node(n1)
#     m.add_node(n2)
#     with pytest.raises(AssertionError) as e:
#         m.finalise()
#     print(e)
#     assert 1 == 2
#
# def test_fail_on_duplicate_names():
#     m = Model("name")
#     n1 = ParameterObserved("a", "a", np.random.random(20))
#     n2 = ParameterObserved("a", "a", np.random.random(10))
#     m.add_node(n1)
#     m.add_node(n2)
#     m.finalise()
#     assert 1 == 2