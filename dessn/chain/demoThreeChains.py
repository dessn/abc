import numpy as np
from dessn.chain.chain import ChainConsumer


class DemoThreeChains:
    """ The multiple chain demo for Chain Consumer. Dummy class used to get documentation caught by ``sphinx-apidoc``,
    it servers no other purpose.

    Running this file in python creates three random data sets, representing three separate chains.

    First, we create a consumer and load the first two chains, and tell it to plot with filled contours.

    The second thing we do is create a different consumer, and load all three chains into it. We also supply the
    parameter labels the first time we load in a chain. The plot for this is saved to the png file below:

    .. figure::     ../dessn/chain/demoThreeChains.png
        :align:     center

    """
    def __init__(self):
        pass

if __name__ == "__main__":
    ndim, nsamples = 3, 200000
    np.random.seed(0)

    data = np.random.randn(nsamples, ndim)
    data[:, 2] += data[:, 1] * data[:, 2]
    data[:, 1] = data[:, 1] * 3 + 5

    data2 = np.random.randn(nsamples, ndim)
    data2[:, 0] -= 1
    data2[:, 2] += data2[:, 1]**2
    data2[:, 1] = data2[:, 1] * 2 - 5

    data3 = np.random.randn(nsamples, ndim)
    data3[:, 2] -= 1
    data3[:, 0] += np.abs(data3[:, 1])
    data3[:, 1] += 2
    data3[:, 1] = data3[:, 2] * 2 - 5

    # You can plot the data directly without worrying about labels like the single chain example
    ChainConsumer().add_chain(data).add_chain(data2).configure_contour(contourf=True).plot(display=True)

    # If you pass in parameter labels and only one chain, you can also get parameter bounds
    c = ChainConsumer()\
        .add_chain(data, parameters=["$x$", "$y$", r"$\epsilon$"], name="Test chain")\
        .add_chain(data2, name="Chain2")\
        .add_chain(data3, name="Chain3")\
        .plot(display=True, filename="demoThreeChains.png")
