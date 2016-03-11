import numpy as np
from chain import ChainConsumer


class DemoVarious(object):
    r""" The demo for various functions and usages of Chain Consumer.

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

        data3 = np.random.randn(nsamples, ndim)
        data3[:, 2] -= 1
        data3[:, 0] += np.abs(data3[:, 1])
        data3[:, 1] += 2
        data3[:, 1] = data3[:, 2] * 2 - 5

        data4 = (data[:] + 1.0) * 1.5

        self.data = data
        self.data2 = data2
        self.data3 = data3
        self.data4 = data4
        self.parameters = ["$x$", "$y$", r"$\alpha$", r"$\beta$"]

    def various1_no_histogram(self):
        """ Plot data without histogram or cloud. For those liking the minimalistic approach

        .. figure::     ../dessn/chain/demoVarious1_NoHist.png
            :align:     center

        """
        c = ChainConsumer().add_chain(self.data, parameters=self.parameters)
        c.configure_general(plot_hists=False).configure_contour(cloud=False)
        c.plot(filename="demoVarious1_NoHist.png")

    def various2_select_parameters(self):
        """ You can chose to only display a select number of parameters.

        .. figure::     ../dessn/chain/demoVarious2_SelectParameters.png
        """
        c = ChainConsumer().add_chain(self.data, parameters=self.parameters)
        c.plot(parameters=self.parameters[:3], filename="demoVarious2_SelectParameters.png")

    def various3_flip_histogram(self):
        """ YWhen you only display two parameters and don't disable histograms, your plot will look different.

        You can suppress this by passing to ``flip=False`` to :func:`ChainConsumer.configure_general`. See the
        commented out line in code for the actual line to disable this.

        The max number of ticks is also modified in this example.

        .. figure::     ../dessn/chain/demoVarious3_Flip.png
            :align:     center
        """
        c = ChainConsumer().add_chain(self.data, parameters=self.parameters)
        # c.configure_general(flip=False, max_ticks=5)
        c.configure_general(max_ticks=10)
        c.plot(parameters=self.parameters[:2], filename="demoVarious3_Flip.png")

    def various4_summaries(self):
        r""" If there is only one chain to analyse, and you only chose to plot a small number of parameters,
        the parameter summary will be shown above the relevent axis. You can set this to always show or always not show
        by using the ``force_summary`` flag. Also, here we demonstrate more :math:`\sigma` levels!

        .. figure::     ../dessn/chain/demoVarious4_ForceSummary.png
            :align:     center
        """
        c = ChainConsumer().add_chain(self.data, parameters=self.parameters)
        c.configure_bar(summary=False).configure_contour(cloud=False, sigmas=np.linspace(0, 3, 10))
        c.plot(filename="demoVarious4_ForceSummary.png")

    def various5_custom_colours(self):
        r""" You can supply custom colours to the plotting. Be warned, if you have more chains than colours, you *will* get a
        rainbow instead!

        Note that, due to colour scaling, you **must** supply custom colours as full six digit hex colours, such as ``#A87B20``.

        As colours get scaled, it is a good idea to pick something neither too light, dark, or saturated.

        In this example, I also force contour filling and set contour filling opactiy to 0.5, so we can see overlap.

        .. figure::     ../dessn/chain/demoVarious5_CustomColours.png
            :align:     center
        """
        c = ChainConsumer().add_chain(self.data, parameters=self.parameters).add_chain(self.data2)
        c.configure_general(colours=["#B32222", "#D1D10D"])
        c.configure_contour(contourf=True, contourf_alpha=0.5)
        c.plot(filename="demoVarious5_CustomColours.png")

    def various6_truth_values(self):
        r""" The reward for scrolling down so far, the first customised argument that will be frequently used; truth values.

        Truth values can be given as a list the same length of the input parameters, or as a dictionary, keyed by the parameters.

        In the code there are two examples. The first, where a list is passed in, and the second, where an incomplete dictionary
        of truth values is passed in. In the second case, customised values for truth line plotting are used.
        The figures are respectively

        .. figure::     ../dessn/chain/demoVarious6_TruthValues.png
            :align:     center

        .. figure::     ../dessn/chain/demoVarious6_TruthValues2.png
            :align:     center
        """
        c = ChainConsumer().add_chain(self.data, parameters=self.parameters)
        c.plot(filename="demoVarious6_TruthValues.png", truth=[0.0, 5.0, 0.0, 0.0])

        # You can also set truth using a dictionary, like below. If you do it this way, you do not need to
        # set truth values for all parameters
        c.configure_truth(color='w', ls=":", alpha=0.5).plot(filename="demoVarious6_TruthValues2.png", truth={"$x$": 0.0, "$y$": 5.0, r"$\beta$": 0.0})

    def various7_rainbow(self):
        r""" An example on using the rainbow with serif fonts and too many bins!

        .. figure::     ../dessn/chain/demoVarious7_Rainbow.png
            :align:     center
        """
        c = ChainConsumer()
        c.add_chain(self.data, name="A")
        c.add_chain(self.data2, name="B")
        c.add_chain(self.data3, name="C")
        c.add_chain(self.data4, name="D")
        c.configure_general(bins=150, serif=True, rainbow=True)
        c.plot(filename="demoVarious7_Rainbow.png")

    def various8_extents(self):
        r""" An example on using customised extents. Similarly to the example for truth values in
        :func:`various7_truth_values`, you can pass a list in, or a dictionary.

        .. figure::     ../dessn/chain/demoVarious8_Extents.png
            :align:     center
        """
        c = ChainConsumer()
        c.add_chain(self.data)
        c.plot(filename="demoVarious8_Extents.png", extents=[(-5, 5), (0, 15), (-3, 3), (-6, 6)])


if __name__ == "__main__":

    demo = DemoVarious()

    # demo.various1_no_histogram()

    # demo.various2_select_parameters()

    # demo.various3_flip_histgram()

    # demo.various4_summaries()

    # demo.various5_custom_colours()

    # demo.various6_truth_values()

    # demo.various7_rainbow()

    # demo.various8_extents()