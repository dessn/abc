import numpy as np
from chain import ChainConsumer


class DemoOneChain:
    """ The single chain demo for Chain Consumer. Dummy class used to get documentation caught by ``sphinx-apidoc``,
    it servers no other purpose.

    Running this file in python creates a random data set, representing a single MCMC chain, such as you might get from ``emcee``.

    First, we create a consumer and load the chain, and tell it to plot the chain without knowing the parameter labels.
    It is set to so that the plot should pop up. To continue running the code, close the plot.

    The second thing we do is create a different consumer, and load the chain into it. We also supply the
    parameter labels. By default, as we only have a single chain, contours are filled, the marginalised
    histograms are shaded, and the best fit parameter bounds are shown as axis titles.

    The plot for this is saved to the png file below:

    .. figure::     ../dessn/chain/demoOneChain.png
        :align:     center

    """
    def __init__(self):
        pass


if __name__ == "__main__":
    ndim, nsamples = 3, 200000
    np.random.seed(1)
    data = np.random.randn(nsamples, ndim)
    data[:, 2] += data[:, 1] * data[:, 2]
    data[:, 1] = data[:, 1] * 3 + 5

    # You can plot the data directly without worrying about labels
    # Code follows the fluent pattern, so you can string calls together
    ChainConsumer().add_chain(data).plot(display=True, contour_kwargs={"cloud": False})

    # If you pass in parameter labels and only one chain, you can also get parameter bounds
    ChainConsumer().add_chain(data, parameters=["$x$", "$y$", r"$\epsilon$"], name="Test chain").plot(filename="demoOneChain.png")
