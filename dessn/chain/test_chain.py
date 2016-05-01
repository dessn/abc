from . import chain
import numpy as np


class TestChain(object):
    np.random.seed(2)
    n = 10000000
    data = np.random.normal(loc=5.0, scale=1.5, size=n)
    data2 = np.random.normal(loc=3, scale=1.0, size=n)
    data_combined = np.vstack((data, data2)).T

    def test_summary1(self):
        tolerance = 1e-2
        consumer = chain.ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure_general(bins=1.2)
        summary = consumer.get_summary()
        actual = np.array(list(summary[0].values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_summary2(self):
        tolerance = 2e-2
        consumer = chain.ChainConsumer()
        consumer.add_chain(self.data_combined, parameters=["a", "b"], name="chain1")
        consumer.add_chain(self.data_combined, parameters=["c", "d"], name="chain2")
        consumer.configure_general(bins=1.6)
        summary = consumer.get_summary()
        k1 = list(summary[0].keys())
        k2 = list(summary[1].keys())
        assert len(k1) == 2
        assert "a" in k1
        assert "b" in k1
        assert len(k2) == 2
        assert "c" in k2
        assert "d" in k2
        expected1 = np.array([3.5, 5.0, 6.5])
        expected2 = np.array([2.0, 3.0, 4.0])
        diff1 = np.abs(expected1 - np.array(list(summary[0]["a"])))
        diff2 = np.abs(expected2 - np.array(list(summary[0]["b"])))
        assert np.all(diff1 < tolerance)
        assert np.all(diff2 < tolerance)

    def test_output_text(self):
        consumer = chain.ChainConsumer()
        consumer.add_chain(self.data, parameters=["a"])
        vals = consumer.get_summary()[0]["a"]
        text = consumer.get_parameter_text(*vals)
        assert text == r"5.0\pm 1.5"






