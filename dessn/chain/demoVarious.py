import numpy as np
from chain import ChainConsumer


class DemoVarious(object):
    r""" The demo for various functions and usages of Chain Consumer.

    Running this file in python creates two random data sets, representing two separate chains, *for two separate models*.

    This file should show some examples of how to use ChainConsumer in more unusual ways with extra customisation.

    The methods of this class should provide context as to what is being done.

    """
    def __init__(self):
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

        self.data = data
        self.data2 = data2
        self.parameters = ["$x$", "$y$", r"$\alpha$", r"$\beta$"]

    def various1_no_histogram(self):
        """ Plot data without histogram or cloud. For those liking the minimalistic approach

        .. figure::     ../dessn/chain/demoVarious1_NoHist.png
            :align:     center

        """
        # Playing around with only one chain
        c = ChainConsumer().add_chain(self.data, parameters=self.parameters)
        c.plot(plot_hists=False, filename="demoVarious1_NoHist.png", contour_kwargs={"cloud": False})

if __name__ == "__main__":

    demo = DemoVarious()

    demo.various1_no_histogram()
