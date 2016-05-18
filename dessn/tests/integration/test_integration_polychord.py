from ...framework.model import Model
from ...framework.parameter import ParameterUnderlying, ParameterObserved
from ...framework.edge import Edge
from ...framework.samplers.polychord import PolyChord
import pytest
import numpy as np
import os
import shutil


class Observed(ParameterObserved):
    def __init__(self):
        super().__init__("obs", "obs", np.array([0.0]))


class Underlying(ParameterUnderlying):
    def __init__(self):
        super().__init__("mean", "mean")

    def get_suggestion_sigma(self, data):
        return 3.0

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


@pytest.mark.skipif(reason="Cannot install PolyChord using travis as it is behind password wall")
def test_fit():
    m = Model("Model")
    m.add_node(Underlying())
    m.add_node(Observed())
    m.add_edge(TheEdge())
    np.random.seed(0)
    outdir = os.path.abspath("dessn/tests/integration/output")
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    sampler = PolyChord(temp_dir=outdir, num_repeats=1000, boost=1000.0)
    consumer = m.fit(sampler)
    print(consumer.chains[0])
    consumer.configure_general(kde=True)
    summary = np.array(consumer.get_summary()[0]["mean"])
    summary[1] = np.mean(m.flat_chain)
    expected = np.array([-1.0, 0.0, 1.0])
    threshold = 0.3
    diff = np.abs(expected - summary)
    print(expected, summary, diff)
    shutil.rmtree(outdir)
    assert np.all(diff < threshold)
