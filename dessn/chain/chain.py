import matplotlib.pyplot as plt
import numpy as np
import logging
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm


class ChainConsumer(object):
    """ A class for consuming chains produced by an MCMC walk

    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.all_colours = ["#1E88E5", "#D32F2F", "#4CAF50", "#673AB7", "#FFC107", "#795548", "#64B5F6", "#8BC34A", "#757575", "#CDDC39"]
        self.chains = []
        self.names = []
        self.parameters = []
        self.all_parameters = []
        self.default_parameters = None
        self._configured_bar = False
        self._configured_contour = False
        self._configured_truth = False
        self._configured_general = False
        self.parameters_contour = {}
        self.parameters_bar = {}
        self.parameters_truth = {}
        self.parameters_general = {}

    def add_chain(self, chain, parameters=None, name=None):
        """ Add a chain to the consumer.

        Parameters
        ----------
        chain : str|ndarray
            The chain to load. Normally a ``numpy.ndarray``, but can also accept a string. If a string is found, it
            interprets the string as a filename and attempts to load it in.
        parameters : list[str], optional
            A list of parameter names, one for each column (dimension) in the chain.
        name : str, optional
            The name of the chain. Used when plotting multiple chains at once.

        Returns
        -------
        ChainConsumer
            Itself, to allow chaining calls.
        """
        assert chain is not None, "You cannot have a chain of None"
        if isinstance(chain, str):
            if chain.endswith("txt"):
                chain = np.loadtxt(chain)
            else:
                chain = np.load(chain)
        self.chains.append(chain)
        self.names.append(name)
        if self.default_parameters is None and parameters is not None:
            self.default_parameters = parameters

        if parameters is None:
            if self.default_parameters is not None:
                assert chain.shape[1] == len(self.default_parameters), "Chain has %d dimensions, but default parameters have %d dimesions" % (chain.shape[1], len(self.default_parameters))
                parameters = self.default_parameters
                self.logger.debug("Adding chain using default parameters")
            else:
                self.logger.debug("Adding chain with no parameter names")
                parameters = [x for x in range(chain.shape[1])]
        else:
            self.logger.debug("Adding chain with defined parameters")
        for p in parameters:
            if p not in self.all_parameters:
                self.all_parameters.append(p)
        self.parameters.append(parameters)
        return self

    def configure_general(self, bins=None, flip=True, rainbow=None, colours=None, serif=True, plot_hists=True):
        """ Configure the general plotting parameters common across the bar and contour plots. If you do not call this explicitly,
        the :func:`plot` method will invoke this method automatically.

        Parameters
        ----------
        bins : int, optional
            The number of bins to use. Overridden by the :func:`plot` method.
        flip : bool, optional
            Set to false if, when plotting only two parameters, you do not want it to rotate the histogram
            so that it is horizontal.
        rainbow : bool, optional
            Set to True to force use of rainbow colours
        colours : list[str(hex)], optional
            Provide a list of colours to use for each chain. If you provide more chains than colours,
            you *will* get the rainbow colour spectrum.
        serif : bool, optional
            Whether to display ticks and labels with serif font.
        plot_hists : bool, optional
            Whether to plot marginalised distributions or not

        """
        assert rainbow is None or colours is None, "You cannot both ask for rainbow colours and then give explicit colours"

        if bins is None:
            bins = self._get_bins()
        else:
            bins = [bins] * len(self.chains)
        self.parameters_general["bins"] = bins

        self.parameters_general["flip"] = flip
        self.parameters_general["serif"] = serif
        self.parameters_general["rainbow"] = rainbow
        self.parameters_general["plot_hists"] = plot_hists
        if colours is None:
            self.parameters_general["colours"] = self.all_colours
        else:
            self.parameters_general["colours"] = colours

        self._configured_general = True

        return self

    def configure_contour(self, sigmas=None, cloud=None, contourf=None, contourf_alpha=1.0):
        """ Configure the default variables for the contour plots. If you do not call this explicitly,
        the :func:`plot` method will invoke this method automatically.

        Please ensure that you call this method after adding all the relevant datae to the chain consumer,
        as the consume changes configuration values depending on the presupplied data.

        Parameters
        ----------
        sigmas : np.array, optional
            The :math:`\sigma` contour levels to plot. Defaults to [0.5, 1, 2, 3]. Number of contours shown
            decreases with the number of chains to show.
        cloud : bool, optional
            If set, overrides the default behaviour and plots the cloud or not
        contourf : bool, optional
            If set, overrides the default behaviour and plots filled contours or not
        contourf_alpha : float, optional
            Filled contour alpha value override.
        """
        num_chains = len(self.chains)

        if sigmas is None:
            if num_chains == 1:
                sigmas = np.array([0, 0.5, 1, 1.5, 2])
            elif num_chains < 4:
                sigmas = np.array([0, 0.5, 1, 2])
            else:
                sigmas = np.array([0, 1, 2])
        sigmas = np.sort(sigmas)
        self.parameters_contour["sigmas"] = sigmas
        if cloud is None:
            cloud = num_chains == 1
        self.parameters_contour["cloud"] = cloud

        if contourf is None:
            contourf = num_chains == 1
        self.parameters_contour["contourf"] = contourf
        self.parameters_contour["contourf_alpha"] = contourf_alpha

        self._configured_contour = True

        return self

    def configure_bar(self, summary=None):
        """ Configure the bar plots showing the marginalised distributions. If you do not call this explicitly,
        the :func:`plot` method will invoke this method automatically.

        summary : bool, optional
            If overridden, sets whether parameter summaries should be set as axis titles.
            Will not work if you have multiple chains
        """
        if summary is not None:
            summary = summary and len(self.chains) == 1
        self.parameters_bar["summary"] = summary

        self._configured_bar = True
        return self

    def configure_truth(self, **kwargs):
        """ Configure the arguments passed to the ``axvline`` and ``axhline`` methods when plotting truth values.
        If you do not call this explicitly, the :func:`plot` method will invoke this method automatically.

        Recommended to set the parameters ``linestyle``, ``color`` and/or ``alpha`` if you want some basic control.

        Default is to use an opaque black dashed line.
        """
        if kwargs.get("ls") is None and kwargs.get("linestyle") is None:
            kwargs["ls"] = "--"
            kwargs["dashes"] = (3, 3)
        if kwargs.get("color") is None:
            kwargs["color"] = "#000000"
        self.parameters_truth = kwargs
        self._configured_truth = True
        return self

    def plot(self, figsize="COLUMN", parameters=None, filename=None, display=False, truth=None):
        """ Plot the chain

        Parameters
        ----------
        figsize : str|tuple(float), optional
            The figure size to generate. Accepts a regular two tuple of size in inches, or one of several key words.
            The default value of ``COLUMN`` creates a figure of appropriate size of insertion into an A4 LaTeX document
            in two-column mode. ``PAGE`` creates a full page width figure. String arguments are not case sensitive.
        parameters : list[str], optional
            If set, only creates a plot for those specific parameters
        filename : str, optional
            If set, saves the figure to this location
        display : bool, optional
            If True, shows the figure using ``plt.show()``.
        truth : list[float] or dict[str], optional
            A list of truth values corresponding to parameters, or a dictionary of truth values indexed by key

        Returns
        -------
        figure
            the matplotlib figure

        """

        if not self._configured_general:
            self.configure_general()
        if not self._configured_bar:
            self.configure_bar()
        if not self._configured_contour:
            self.configure_contour()
        if not self._configured_truth:
            self.configure_truth()

        if isinstance(figsize, str):
            if figsize.upper() == "COLUMN":
                figsize = (5, 5)
            elif figsize.upper() == "PAGE":
                figsize = (10, 10)
            else:
                raise ValueError("Unknown figure size %s" % figsize)

        if parameters is None:
            parameters = self.all_parameters

        assert truth is None or isinstance(truth, dict) or (isinstance(truth, list) and len(truth) == len(parameters)), \
            "Have a list of %d parameters and %d truth values" % (len(parameters), len(truth))

        if truth is not None and isinstance(truth, list):
            truth = {p: t for p, t in zip(parameters, truth)}

        plot_hists = self.parameters_general["plot_hists"]
        flip = (len(parameters) == 2 and plot_hists and self.parameters_general["flip"])

        fig, axes, params1, params2 = self._get_figure(parameters, figsize=figsize, flip=flip)

        num_bins = self._get_bins()
        fit_values = self.get_summary()
        colours = self._get_colours(self.parameters_general["colours"], rainbow=self.parameters_general["rainbow"])
        summary = self.parameters_bar["summary"]
        if summary is None:
            summary = len(parameters) < 5 and len(self.chains) == 1

        for i, p1 in enumerate(params1):
            for j, p2 in enumerate(params2):
                if i < j:
                    continue
                ax = axes[i, j]
                do_flip = (flip and i == len(params1) - 1)
                if plot_hists and i == j:
                    max_val = None
                    for chain, parameters, colour, bins, fit in zip(self.chains, self.parameters, colours, num_bins, fit_values):
                        if p1 not in parameters:
                            continue
                        index = parameters.index(p1)
                        m = self._plot_bars(ax, p1, chain[:, index], colour, bins=bins, fit_values=fit[p1], flip=do_flip, summary=summary, truth=truth)
                        if max_val is None or m > max_val:
                            max_val = m
                    if do_flip:
                        ax.set_xlim(0, 1.1 * max_val)
                    else:
                        ax.set_ylim(0, 1.1 * max_val)

                else:
                    for chain, parameters, bins, colour, fit in zip(self.chains, self.parameters, num_bins, colours, fit_values):
                        if p1 not in parameters or p2 not in parameters:
                            continue
                        i1 = parameters.index(p1)
                        i2 = parameters.index(p2)
                        self._plot_contour(ax, chain[:, i2], chain[:, i1], p1, p2, colour, bins=bins, fit_values=fit, truth=truth)

        if self.names is not None:
            ax = axes[0, -1]
            artists = [plt.Line2D((0, 1), (0, 0), color=c) for n,c in zip(self.names, colours) if n is not None]
            ax.legend(artists, self.names, loc="center", frameon=False)

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.05)
        if display:
            plt.show()
        return fig

    def _plot_bars(self, ax, parameter, chain_row, colour, bins=25, flip=False, summary=False, fit_values=None, truth=None):

        hist, edges = np.histogram(chain_row, bins=bins, normed=True)
        edge_center = 0.5 * (edges[:-1] + edges[1:])
        if flip:
            orientation = "horizontal"
        else:
            orientation = "vertical"
        ax.hist(edge_center, weights=hist, bins=edges, histtype="step", color=colour, orientation=orientation)
        interpolator = interp1d(edge_center, hist, kind="nearest")
        if len(self.chains) == 1 and fit_values is not None:
            lower = fit_values[0]
            upper = fit_values[2]
            if lower is not None and upper is not None:
                x = np.linspace(lower, upper, 1000)
                if flip:
                    ax.fill_betweenx(x, np.zeros(x.shape), interpolator(x), color=colour, alpha=0.2)
                else:
                    ax.fill_between(x, np.zeros(x.shape), interpolator(x), color=colour, alpha=0.2)
                if summary and isinstance(parameter, str):
                    ax.set_title(r"$%s = %s$" % (parameter.strip("$"), self.get_parameter_text(*fit_values)), fontsize=14)
        if truth is not None:
            truth_value = truth.get(parameter)
            if truth_value is not None:
                if flip:
                    ax.axhline(truth_value, **self.parameters_truth)
                else:
                    ax.axvline(truth_value, **self.parameters_truth)
        return hist.max()

    def _plot_contour(self, ax, x, y, px, py, colour, bins=25, fit_values=None, truth=None):

        levels = 1.0 - np.exp(-0.5 * self.parameters_contour["sigmas"] ** 2)

        colours = self._scale_colours(colour, len(levels))
        colours2 = [self._scale_colour(colours[0], 0.7)] + [self._scale_colour(c, 0.8) for c in colours[:-1]]

        hist, x_bins, y_bins = np.histogram2d(x, y, bins=bins)
        x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
        y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
        hist[hist == 0] = 1E-16
        vals = self._convert_to_stdev(hist.T)
        if self.parameters_contour["cloud"]:
            skip = x.size / 50000
            ax.scatter(x[::skip], y[::skip], s=10, alpha=0.2, c=colours[1], marker=".", edgecolors="none")
        if self.parameters_contour["contourf"]:
            ax.contourf(x_centers, y_centers, vals, levels=levels, colors=colours, alpha=self.parameters_contour["contourf_alpha"])
        ax.contour(x_centers, y_centers, vals, levels=levels, colors=colours2)

        if truth is not None:
            truth_value = truth.get(px)
            if truth_value is not None:
                ax.axhline(truth_value, **self.parameters_truth)
            truth_value = truth.get(py)
            if truth_value is not None:
                ax.axvline(truth_value, **self.parameters_truth)

    def get_summary(self):
        """  Gets a summary of the marginalised parameter distributions.

        Returns
        -------
        list of dictionaries
            One entry per chain, parameter bounds stored in dictionary with parameter as key
        """
        results = []
        for chain, parameters in zip(self.chains, self.parameters):
            res = {}
            for i, p in enumerate(parameters):
                summary = self._get_parameter_summary(chain[:, i], p)
                res[p] = summary
            results.append(res)
        return results

    def get_parameter_text(self, lower, maximum, upper):
        """ Generates LaTeX appropriate text from marginalised parameter bounds.

        Parameters
        ----------
        lower : float
            The lower bound on the parameter
        maximum : float
            The value of the parameter with maximum probability
        upper : float
            The upper bound on the parameter
        """
        if lower is None or upper is None:
            return ""
        upper_error = upper - maximum
        lower_error = maximum - lower
        resolution = min(np.floor(np.log10(np.abs(upper_error))), np.floor(np.log10(np.abs(lower_error))))
        factor = 0
        if np.abs(resolution) > 3:
            factor = -resolution - 1
        upper_error *= 10 ** factor
        lower_error *= 10 ** factor
        maximum *= 10 ** factor
        upper_error = round(upper_error, 1)
        lower_error = round(lower_error, 1)
        maximum = round(maximum, 1)
        if maximum == -0.0:
            maximum = 0.0
        upper_error_text = "%0.1f" % upper_error
        lower_error_text = "%0.1f" % lower_error
        if upper_error_text == lower_error_text:
            text = r"%0.1f\pm %s" % (maximum, lower_error_text)
        else:
            text = r"%0.1f^{+%s}_{-%s}" % (maximum, upper_error_text, lower_error_text)
        if factor != 0:
            text = r"\left( %s \right) \times 10^{%d}" % (text, -factor)
        return text

    def _get_colours(self, colours, rainbow=False):
        num_chains = len(self.chains)
        if rainbow or num_chains > len(colours):
            colours = cm.rainbow(np.linspace(0, 1, num_chains))
        else:
            colours = colours[:num_chains]
        return colours

    def _get_figure(self, all_parameters, flip, figsize=(5, 5), max_ticks=5):
        n = len(all_parameters)
        plot_hists = self.parameters_general["plot_hists"]
        if not plot_hists:
            n -= 1

        if n == 2 and plot_hists and flip:
            gridspec_kw = {'width_ratios': [3, 1], 'height_ratios': [1, 3]}
        else:
            gridspec_kw = {}
        fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False, gridspec_kw=gridspec_kw)

        if self.parameters_general["serif"]:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.05, hspace=0.05)
        
        extents = {}
        for p in all_parameters:
            min_val = None
            max_val = None
            for chain, parameters in zip(self.chains, self.parameters):
                if p not in parameters:
                    continue
                index = parameters.index(p)
                mean = np.mean(chain[:, index])
                std = np.std(chain[:, index])
                min_prop = mean - 3 * std
                max_prop = mean + 3 * std
                if min_val is None or min_prop < min_val:
                    min_val = min_prop
                if max_val is None or max_prop > max_val:
                    max_val = max_prop
            extents[p] = (min_val, max_val)

        if plot_hists:
            params1 = all_parameters
            params2 = all_parameters
        else:
            params1 = all_parameters[1:]
            params2 = all_parameters[:-1]
        for i, p1 in enumerate(params1):
            for j, p2 in enumerate(params2):
                ax = axes[i, j]
                display_x_ticks = False
                display_y_ticks = False
                if i < j:
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    if i != n - 1 or (flip and j == n - 1):
                        ax.set_xticks([])
                    else:
                        display_x_ticks = True
                        if isinstance(p2, str):
                            ax.set_xlabel(p2, fontsize=14)
                    if j != 0 or (plot_hists and i == 0):
                        ax.set_yticks([])
                    else:
                        display_y_ticks = True
                        if isinstance(p1, str):
                            ax.set_ylabel(p1, fontsize=14)
                    if display_x_ticks:
                        [l.set_rotation(45) for l in ax.get_xticklabels()]
                        ax.xaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                    if display_y_ticks:
                        [l.set_rotation(45) for l in ax.get_yticklabels()]
                        ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                    if i != j or not plot_hists:
                        ax.set_ylim(extents[p1])
                    ax.set_xlim(extents[p2])

        return fig, axes, params1, params2

    def _get_bins(self):
        proposal = [np.floor(0.1 * np.sqrt(chain.shape[0])) for chain in self.chains]
        return proposal
        
    def _clamp(self, val, minimum=0, maximum=255):
        if val < minimum:
            return minimum
        if val > maximum:
            return maximum
        return val
    
    def _scale_colours(self, colour, num):
        # http://thadeusb.com/weblog/2010/10/10/python_scale_hex_color
        scales = np.logspace(np.log(0.7), np.log(1.4), num)
        colours = [self._scale_colour(colour, scale) for scale in scales]
        return colours

    def _scale_colour(self, colour, scalefactor):
        hex = colour.strip('#')
        if scalefactor < 0 or len(hex) != 6:
            return hex

        r, g, b = int(hex[:2], 16), int(hex[2:4], 16), int(hex[4:], 16)
        r = self._clamp(r * scalefactor)
        g = self._clamp(g * scalefactor)
        b = self._clamp(b * scalefactor)
        return "#%02x%02x%02x" % (r, g, b)

    def _convert_to_stdev(self, sigma):
        # From astroML
        shape = sigma.shape
        sigma = sigma.ravel()
        i_sort = np.argsort(sigma)[::-1]
        i_unsort = np.argsort(i_sort)
    
        sigma_cumsum = 1.0* sigma[i_sort].cumsum()
        sigma_cumsum /= sigma_cumsum[-1]
    
        return sigma_cumsum[i_unsort].reshape(shape)

    def _get_parameter_summary(self, data, parameter, bins=50, desired_area=0.6827):
        hist, edges = np.histogram(data, bins=bins, normed=True)
        edge_centers = 0.5 * (edges[1:] + edges[:-1])
        
        xs = np.linspace(edge_centers[0], edge_centers[-1], 10000)
        ys = interp1d(edge_centers, hist, kind="linear")(xs)
        
        cs = ys.cumsum()
        cs /= cs.max()
        
        startIndex = ys.argmax()
        maxVal = ys[startIndex]
        minVal = 0
        threshold = 0.001

        x1 = None
        x2 = None
        count = 0
        while x1 is None:
            mid = (maxVal + minVal) / 2.0
            count += 1                    
            try:
                if count > 50:
                    raise Exception("Failed to converge")
                i1 = np.where(ys[:startIndex] > mid)[0][0]
                i2 = startIndex + np.where(ys[startIndex:] < mid)[0][0]
                area = cs[i2] - cs[i1]
                deviation = np.abs(area - desired_area)
                if deviation < threshold:
                    x1 = xs[i1]
                    x2 = xs[i2]
                elif area < desired_area:
                    maxVal = mid
                elif area > desired_area:
                    minVal = mid
            except:
                self.logger.warn("Parameter %s is not constrained" % parameter)
                return [None, xs[startIndex], None]

        return [x1, xs[startIndex], x2]
        

