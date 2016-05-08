from ...framework.edge import EdgeTransformation, Edge
import pytest


def test_edge_assertion_fails1():
    with pytest.raises(AssertionError):
        Edge("name1", 2)


def test_edge_assertion_fails2():
    with pytest.raises(AssertionError):
        Edge(2, "name2")


def test_edge_not_implemented():
    class E1(Edge):
        def __init__(self):
            super().__init__("a", "b")

    e = E1()
    with pytest.raises(NotImplementedError):
        e.get_log_likelihood(None)


def test_edge_transformation_not_implemented():
    class E2(EdgeTransformation):
        def __init__(self):
            super().__init__("a", "b")

    e = E2()
    with pytest.raises(NotImplementedError):
        e.get_transformation(None)


def test_edge_transformation_likelihood_failure():
    class E3(EdgeTransformation):
        def __init__(self):
            super().__init__("a", "b")
    e = E3()
    with pytest.raises(PermissionError):
        e.get_log_likelihood({})
