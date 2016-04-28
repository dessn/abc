import numpy as np
from .chain import ChainConsumer


class DemoTable:
    r""" The multiple chain demo for Chain Consumer. Dummy class used to get documentation caught by ``sphinx-apidoc``,
    it servers no other purpose.

    Running this file in python creates two random data sets, representing two separate chains, *for two separate models*.

    This example shows the output of calling the :func:`~dessn.chain.chain.ChainConsumer.get_latex_table` method.

    .. figure::     ../dessn/chain/demoTable.png
        :align:     center

    """
    def __init__(self):
        pass

if __name__ == "__main__":
    ndim, nsamples = 4, 200000
    np.random.seed(0)

    data = np.random.randn(nsamples, ndim)
    data[:, 2] += data[:, 1] * data[:, 2]
    data[:, 1] = data[:, 1] * 3 + 5
    data[:, 3] /= (np.abs(data[:, 1]) + 1)

    data2 = np.random.randn(nsamples, ndim)
    data2[:, 0] -= 1
    data2[:, 2] += data2[:, 1]**2
    data2[:, 1] = data2[:, 1] * 2 - 5
    data2[:, 3] = data2[:, 3] * 1.5 + 2

    # If you pass in parameter labels and only one chain, you can also get parameter bounds
    c = ChainConsumer()
    c.add_chain(data, parameters=["$x$", "$y$", r"$\alpha$", r"$\beta$"], name="Model A")
    c.add_chain(data2, parameters=["$x$", "$y$", r"$\alpha$", r"$\gamma$"], name="Model B")
    table = c.get_latex_table(caption="The maximum likelihood results for the tested models", label="tab:example")
    print(table)