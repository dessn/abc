from . import chain
import numpy as np
import tempfile
import os


class TestChain(object):
    np.random.seed(1)
    n = 1000000
    data = np.random.normal(loc=5.0, scale=1.5, size=n)
    data2 = np.random.normal(loc=3, scale=1.0, size=n)
    data_combined = np.vstack((data, data2)).T

    def test_summary(self):
        tolerance = 5e-2
        consumer = chain.ChainConsumer()
        consumer.add_chain(self.data[::2])
        consumer.configure_general(kde=True)
        summary = consumer.get_summary()
        actual = np.array(list(summary[0].values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_summary2(self):
        tolerance = 5e-2
        consumer = chain.ChainConsumer()
        consumer.add_chain(self.data_combined, parameters=["a", "b"], name="chain1")
        consumer.add_chain(self.data_combined, name="chain2")
        consumer.configure_general(bins=1.9)
        summary = consumer.get_summary()
        k1 = list(summary[0].keys())
        k2 = list(summary[1].keys())
        assert len(k1) == 2
        assert "a" in k1
        assert "b" in k1
        assert len(k2) == 2
        assert "a" in k2
        assert "b" in k2
        expected1 = np.array([3.5, 5.0, 6.5])
        expected2 = np.array([2.0, 3.0, 4.0])
        diff1 = np.abs(expected1 - np.array(list(summary[0]["a"])))
        diff2 = np.abs(expected2 - np.array(list(summary[0]["b"])))
        assert np.all(diff1 < tolerance)
        assert np.all(diff2 < tolerance)

    def test_summary1(self):
        tolerance = 5e-2
        consumer = chain.ChainConsumer()
        consumer.add_chain(self.data)
        consumer.configure_general(bins=1.6)
        summary = consumer.get_summary()
        actual = np.array(list(summary[0].values())[0])
        expected = np.array([3.5, 5.0, 6.5])
        diff = np.abs(expected - actual)
        assert np.all(diff < tolerance)

    def test_output_text(self):
        consumer = chain.ChainConsumer()
        consumer.add_chain(self.data, parameters=["a"])
        vals = consumer.get_summary()[0]["a"]
        text = consumer.get_parameter_text(*vals)
        assert text == r"5.0\pm 1.5"

    def test_output_text_asymmetric(self):
        p1 = [1.0, 2.0, 3.5]
        consumer = chain.ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"2.0^{+1.5}_{-1.0}"

    def test_output_format1(self):
        p1 = [1.0e-1, 2.0e-1, 3.5e-1]
        consumer = chain.ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"0.20^{+0.15}_{-0.10}"

    def test_output_format2(self):
        p1 = [1.0e-2, 2.0e-2, 3.5e-2]
        consumer = chain.ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"0.020^{+0.015}_{-0.010}"

    def test_output_format3(self):
        p1 = [1.0e-3, 2.0e-3, 3.5e-3]
        consumer = chain.ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"\left( 2.0^{+1.5}_{-1.0} \right) \times 10^{-3}"

    def test_output_format4(self):
        p1 = [1.0e3, 2.0e3, 3.5e3]
        consumer = chain.ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"\left( 2.0^{+1.5}_{-1.0} \right) \times 10^{3}"

    def test_output_format5(self):
        p1 = [1.1e6, 2.2e6, 3.3e6]
        consumer = chain.ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"\left( 2.2\pm 1.1 \right) \times 10^{6}"

    def test_output_format6(self):
        p1 = [1.0e-2, 2.0e-2, 3.5e-2]
        consumer = chain.ChainConsumer()
        text = consumer.get_parameter_text(*p1, wrap=True)
        assert text == r"$0.020^{+0.015}_{-0.010}$"

    def test_output_format7(self):
        p1 = [None, 2.0e-2, 3.5e-2]
        consumer = chain.ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == ""

    def test_output_format8(self):
        p1 = [-1, -0.0, 1]
        consumer = chain.ChainConsumer()
        text = consumer.get_parameter_text(*p1)
        assert text == r"0.0\pm 1.0"

    def test_file_loading1(self):
        data = self.data[:1000]
        directory = tempfile._get_default_tempdir()
        filename = next(tempfile._get_candidate_names())
        filename = directory + os.sep + filename + ".txt"
        np.savetxt(filename, data)
        consumer = chain.ChainConsumer()
        consumer.add_chain(filename)
        summary = consumer.get_summary()
        actual = np.array(list(summary[0].values())[0])
        assert np.abs(actual[1] - 5.0) < 0.5

    def test_file_loading2(self):
        data = self.data[:1000]
        directory = tempfile._get_default_tempdir()
        filename = next(tempfile._get_candidate_names())
        filename = directory + os.sep + filename + ".npy"
        np.save(filename, data)
        consumer = chain.ChainConsumer()
        consumer.add_chain(filename)
        summary = consumer.get_summary()
        actual = np.array(list(summary[0].values())[0])
        assert np.abs(actual[1] - 5.0) < 0.5

    def test_convergence_failure(self):
        data = np.concatenate((np.random.normal(loc=0.0, size=10000),
                              np.random.normal(loc=4.0, size=10000)))
        consumer = chain.ChainConsumer()
        consumer.add_chain(data)
        summary = consumer.get_summary()
        actual = np.array(list(summary[0].values())[0])
        consumer.plot(filename="death")
        print(actual)
        assert actual[0] is None and actual[2] is None
