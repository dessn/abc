from ...framework.parameter import ParameterObserved, ParameterUnderlying, \
    ParameterDiscrete, Parameter
import pytest
import numpy as np


def test_parameter_assertions():
    class P(Parameter):
        def __init__(self, *args):
            super().__init__(*args)

    with pytest.raises(AssertionError):
        p = P("string", 1)

    with pytest.raises(AssertionError):
        p = P(1, "string")

    p = P("string", "string")


def test_parameter_observed():
    class P(ParameterObserved):
        def __init__(self, data):
            super().__init__("name", "label", data)
    d = np.arange(5)
    p = P(d)
    assert p.get_data() == {"name": d}


def test_parameter_underlying():
    class P(ParameterUnderlying):
        def __init__(self):
            super().__init__("name", "label")
    p = P()
    assert p.get_suggestion_requirements() == []


def test_parameter_discrete():
    class P(ParameterDiscrete):
        def __init__(self):
            super().__init__("name", "label")
    p = P()
    with pytest.raises(NotImplementedError):
        p.get_discrete({})
    assert p.get_discrete_requirements() == []
