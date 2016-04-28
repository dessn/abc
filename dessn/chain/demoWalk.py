import numpy as np
from .chain import ChainConsumer


class DemoWalk:
    """ The single chain demo for Chain Consumer. Dummy class used to get documentation caught by ``sphinx-apidoc``,
    it servers no other purpose.

    Running this file in python creates a random data set, representing a single MCMC chain, such as you might get from ``emcee``.

    We want to see if our walks are behaving as expected, which means we should see them scattered around the underlying
    truth value (or actual truth value if supplied). This is a good consistency check, because if the burn in period
    (which should have already been removed from the chain) was insufficiently long, you would expect to see tails
    appear in this plot.

    Because individual samples can be noisy, there is also the option to pass an integer via parameter ``convolve``, which
    overplots a boxcar smoothed version of the steps, where ``convolve`` sets the smooth window size.

    The plot for this is saved to the png file below:

    .. figure::     ../dessn/chain/demoWalks.png
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

    ps = ["$x$", "$y$", r"$\epsilon$"]
    ChainConsumer().add_chain(data, parameters=ps).plot_walks(filename="demoWalks.png", truth=[0, 5, 0], convolve=100)
