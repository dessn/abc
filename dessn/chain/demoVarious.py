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
        c = ChainConsumer().add_chain(self.data, parameters=self.parameters)
        c.plot(plot_hists=False, filename="demoVarious1_NoHist.png", contour_kwargs={"cloud": False})

    def various2_select_parameters(self):
        """ You can chose to only display a select number of parameters.

        .. figure::     ../dessn/chain/demoVarious2_SelectParameters.png
        """
        c = ChainConsumer().add_chain(self.data, parameters=self.parameters)
        c.plot(parameters=self.parameters[:2], filename="demoVarious2_SelectParameters.png", contour_kwargs={"cloud": False})

    def various3_flip_histgram(self):
        """ YWhen you only display two parameters and don't disable histograms, your plot will look different.

        You can suppress this by passing to plot ``dont_flip=True``.

        .. figure::     ../dessn/chain/demoVarious3_Flip.png
            :align:     center
        """
        c = ChainConsumer().add_chain(self.data, parameters=self.parameters)
        c.plot(parameters=self.parameters[:2], filename="demoVarious3_Flip.png", contour_kwargs={"cloud": False})

    def various4_summaries(self):
        r""" If there is only one chain to analyse, and you only chose to plot a small number of parameters,
        the parameter summary will be shown above the relevent axis. You can set this to always show or always not show
        by using the ``force_summary`` flag. Also, here we demonstrate more :math:`\sigma` levels!

        .. figure::     ../dessn/chain/demoVarious4_ForceSummary.png
            :align:     center
        """
        c = ChainConsumer().add_chain(self.data, parameters=self.parameters)
        c.plot(filename="demoVarious4_ForceSummary.png", force_summary=False, contour_kwargs={"cloud": False, "sigmas": np.linspace(0, 3, 10)})

if __name__ == "__main__":

    demo = DemoVarious()

    # demo.various1_no_histogram()

    # demo.various2_select_parameters()

    # demo.various3_flip_histgram()

    demo.various4_summaries()