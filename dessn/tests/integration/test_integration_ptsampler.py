from ...framework.model import Model
from ...framework.parameter import ParameterUnderlying, ParameterObserved
from ...framework.edge import Edge
from ...framework.samplers.tempered import ParallelTemperedSampler
import numpy as np


class Observed(ParameterObserved):
    def __init__(self):
        super().__init__("obs", "obs", np.array([0.0]))


class Underlying(ParameterUnderlying):
    def __init__(self):
        super().__init__("mean", "mean")

    def get_suggestion_sigma(self, data):
        return 2.0

    def get_suggestion(self, data):
        return 0.0

    def get_log_prior(self, data):
        return 0.0


class TheEdge(Edge):
    def __init__(self):
        super().__init__("obs", "mean")

    def get_log_likelihood(self, data):
        o, m = data["obs"], data["mean"]
        return -0.5 * (o - m)**2 - np.log(np.sqrt(2 * np.pi) * 1.0)


def test_fit():
    m = Model("Model")
    m.add_node(Underlying())
    m.add_node(Observed())
    m.add_edge(TheEdge())
    np.random.seed(0)
    sampler = ParallelTemperedSampler(num_steps=4600, num_burn=600)
    m.fit(sampler)
    consumer = m.get_consumer()
    consumer.configure_general(kde=True)
    summary = np.array(consumer.get_summary()[0]["mean"])
    summary[1] = np.mean(m.flat_chain)
    expected = np.array([-1.0, 0.0, 1.0])
    threshold = 0.1
    diff = np.abs(expected - summary)
    assert np.all(diff < threshold)
