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

    def _get_colours(self, rainbow=False):
        num_chains = len(self.chains)
        if rainbow or num_chains > 5:
            colours = cm.rainbow(np.linspace(0, 1, num_chains))
        else:
            colours = self.all_colours[:num_chains]
        return colours

    def _get_figure(self, all_parameters, figsize=(5, 5), max_ticks=5, serif=True, plot_hists=True):
        n = len(all_parameters)
        if not plot_hists:
            n -= 1
        fig, axes = plt.subplots(n, n, figsize=figsize)
        if not isinstance(axes, list) and not isinstance(axes, np.ndarray):
            self.logger.debug("Only one axis generated")
            axes = np.array([[axes]])
        if serif:
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
                    if i != n - 1:
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

    def plot(self, figsize="COLUMN", parameters=None, filename=None, display=False, rainbow=False, serif=True, contour_kwargs=None, plot_hists=True):
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
        rainbow : bool, optional
            If true, forces the use of rainbow colours when displaying multiple chains. By default, under a certain
            number of chains to show, this method uses a predefined list of colours.
        serif : bool, optional
            Sets all plot text to serif.
        contour_kwargs : dict, optional
            A dictionary of optional arguments to pass to the :func:`plot_contour` function.
        plot_hists : bool, optional
            Whether to plot histograms or not

        Returns
        -------
        figure
            the matplotlib figure

        """
        if contour_kwargs is None:
            contour_kwargs = {}
        if isinstance(figsize, str):
            if figsize.upper() == "COLUMN":
                figsize = (5, 5)
            elif figsize.upper() == "PAGE":
                figsize = (10, 10)
            else:
                raise ValueError("Unknown figure size %s" % figsize)

        if parameters is None:
            parameters = self.all_parameters
        fig, axes, params1, params2 = self._get_figure(parameters, figsize=figsize, serif=serif, plot_hists=plot_hists)

        num_bins = self._get_bins()
        fit_values = self.get_summary()
        colours = self._get_colours(rainbow=rainbow)
        for i, p1 in enumerate(params1):
            for j, p2 in enumerate(params2):
                if i < j:
                    continue
                ax = axes[i, j]
                if plot_hists and i == j:
                    max_val = None
                    for chain, parameters, colour, bins, fit in zip(self.chains, self.parameters, colours, num_bins, fit_values):
                        if p1 not in parameters:
                            continue
                        index = parameters.index(p1)
                        m = self.plot_bars(ax, p1, chain[:, index], colour=colour, bins=bins, fit_values=fit[p1])
                        if max_val is None or m > max_val:
                            max_val = m
                    ax.set_ylim(0, 1.1 * max_val)
                    
                else:
                    for chain, parameters, bins, colour, fit in zip(self.chains, self.parameters, num_bins, colours, fit_values):
                        if p1 not in parameters or p2 not in parameters:
                            continue
                        i1 = parameters.index(p1)
                        i2 = parameters.index(p2)
                        self.plot_contour(ax, chain[:, i2], chain[:, i1], colour=colour, bins=bins, fit_values=fit, **contour_kwargs)

        if self.names is not None:
            ax = axes[0, -1]
            artists = [plt.Line2D((0,1),(0,0), color=c) for n,c in zip(self.names, colours) if n is not None]
            ax.legend(artists, self.names, loc="center", frameon=False)

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.05)
        if display:
            plt.show()
        return fig

    def plot_bars(self, ax, parameter, chain_row, bins=50, colour='#222222', fit_values=None):
        """ Method responsible for plotting the marginalised distributions

        Parameters
        ----------
        ax : matplotlib axis
            Upon which the plot is drawn
        parameter : str
            The parameter label, if it exists
        chain_row : np.ndarray
            The data corresponding to the parameter
        bins : int, optional
            The number of bins to use. Default value is overridden by :func:`plot`
        colour : str
            The colour to use when plotting. Default value is overridden

        Returns
        -------
        float
            the maximum value of the histogram plot (used to ensure vertical spacing)
        """
        hist, edges = np.histogram(chain_row, bins=bins, normed=True)
        edge_center = 0.5 * (edges[:-1] + edges[1:])
        ax.hist(edge_center, weights=hist, bins=edges, histtype="step", color=colour)
        interpolator = interp1d(edge_center, hist, kind="nearest")
        if len(self.chains) == 1 and fit_values is not None:
            lower = fit_values[0]
            upper = fit_values[2]
            if lower is not None and upper is not None:
                x = np.linspace(lower, upper, 1000)
                ax.fill_between(x, np.zeros(x.shape), interpolator(x), color=colour, alpha=0.2)
                if isinstance(parameter, str):
                    ax.set_title(r"$%s = %s$" % (parameter.strip("$"), self.get_parameter_text(*fit_values)), fontsize=14)
        return hist.max()
        
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
        
    def plot_contour(self, ax, x, y, bins=50, sigmas=None, colour='#222222', fit_values=None, force_contourf=False, cloud=True):
        r""" Plots contours of the probability surface between two parameters

        Parameters
        ----------
        ax : figure.axis
            The axis to plot to
        x : np.ndarray
            The ``x`` axis array of data
        y : np.ndarray
            The ``y`` axis array of data
        bins : int, optional
            The number of bins to use. Overridden by the :func:`plot` method.
        sigmas : np.array, optional
            The :math:`\sigma` contour levels to plot. Defaults to [0.5, 1, 2, 3]. Number of contours shown
            decreases with the number of chains to show.
        colour : str(hex code), optional
            The colour to plot the contours in. Overridden by the :func:`plot` method.
        fit_values : np.array, optional
            An array representing the lower bound, maximum, and upper bound of the marignliased parameters
        force_contourf : bool
            Can force the plotting method to plot filled contours even when it would normally be disabled.
            It is normally disabled when plotting multiple chains.
        cloud : bool
            If true and there is only one chain, plots the cloud of points. If false, or there are more than
            one chain, does not plot the cloud.
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
        levels = 1.0 - np.exp(-0.5 * sigmas ** 2)

        colours = self._scale_colours(colour, len(levels))
        colours2 = [self._scale_colour(colours[0], 0.7)] + [self._scale_colour(c, 0.8) for c in colours[:-1]]

        hist, x_bins, y_bins = np.histogram2d(x, y, bins=bins)
        x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
        y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
        hist[hist == 0] = 1E-16
        vals = self._convert_to_stdev(hist.T)
        if cloud and num_chains == 1:
            skip = x.size / 50000
            ax.scatter(x[::skip], y[::skip], s=10, alpha=0.2, c=colours[1], marker=".", edgecolors="none")
        if len(self.chains) == 1 or force_contourf:
            cf = ax.contourf(x_centers, y_centers, vals, levels=levels, colors=colours, alpha=1.0)
        c = ax.contour(x_centers, y_centers, vals, levels=levels, colors=colours2)

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
