import matplotlib.pyplot as plt
import numpy as np
import logging
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator


class ChainConsumer(object):
    def __init__(self, chains, parameters=None, names=None):
        self.logger = logging.getLogger(__name__)
        self.all_colours = ["#1E88E5", "#D32F2F", "#4CAF50", "#673AB7", "#FFC107", "#795548", "#64B5F6", "#8BC34A", "#757575", "#CDDC39"]
        self.format_data(chains, parameters, names=names)
        self.round_to_n = lambda x, n: round(x, -int(np.floor(np.log10(x))) + (n - 1))

    def format_data(self, chains, parameters, names=None):
        if chains is None:
            raise ValueError("You cannot have a chain of None")
        if isinstance(chains, np.ndarray) or isinstance(chains, str):
            logging.info("Single chain and parameters")
            chains = [chains]
            parameters = [parameters]
        else:
            assert len(chains) == len(names), "Please ensure you have the same number of names as you do chains"
            logging.info("Found %d chains" % len(chains))
    
        for i, chain in enumerate(chains):
            if isinstance(chain, str):
                if chain.endswith("txt"):
                    chains[i] = np.loadtxt(chain)
                else:
                    chains[i] = np.load(chain)
        if parameters[0] is None or isinstance(parameters[0], str):
            num_parameters = [chain.shape[1] for chain in chains]
            assert np.unique(num_parameters).size == 1, "If you don't specific parameters for each chain, your chains have to have identical dimensionality"
            if parameters[0] is None:
                parameters = [list(np.arange(chain.shape[1])) for chain in chains]
            else:
                parameters = [parameters for chain in chains]
        elif len(chains) > 1:
            for chain, params in zip(chains, parameters):
                assert chain.shape[1] == len(params), "Chain has dimension %d but received %d parameters" % (chain.shape[1], len(params))
    
        all_parameters = []
        for p in parameters:

            for val in p:
                if val not in all_parameters:
                    all_parameters.append(val)
            
            
        self.all_parameters = all_parameters
        self.parameters = parameters
        self.chains = chains
        self.names = names
        self.colours = self.all_colours[:len(self.chains)]
        
    def get_figure(self, figsize=(5, 5), max_ticks=5):
        n = len(self.all_parameters)
        fig, axes = plt.subplots(n, n, figsize=figsize)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.05, hspace=0.05)
        
        extents = []
        for p in self.all_parameters:
            min_val = None
            max_val = None
            for chain, parameters in zip(self.chains, self.parameters):
                index = parameters.index(p)
                mean = np.mean(chain[:, index])
                std = np.std(chain[:, index])
                min_prop = mean - 3 * std
                max_prop = mean + 3 * std
                if min_val is None or min_prop < min_val:
                    min_val = min_prop
                if max_val is None or max_prop > max_val:
                    max_val = max_prop
            extents.append((min_val, max_val))
            
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                display_x_ticks = False
                display_y_ticks = False
                if i < j:
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                if i != n - 1:
                    ax.set_xticks([])
                else:
                    display_x_ticks = True
                    if isinstance(self.all_parameters[j], str):
                        ax.set_xlabel(self.all_parameters[j], fontsize=14)
                if j != 0 or i == 0:
                    ax.set_yticks([])
                else:
                    display_y_ticks = True
                    if isinstance(self.all_parameters[i], str):
                        ax.set_ylabel(self.all_parameters[i], fontsize=14)
                if display_x_ticks:
                    [l.set_rotation(45) for l in ax.get_xticklabels()]
                    ax.xaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                if display_y_ticks:
                    [l.set_rotation(45) for l in ax.get_yticklabels()]
                    ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                if i != j:
                    ax.set_ylim(extents[i])
                ax.set_xlim(extents[j])

        return fig, axes

    def get_bins(self):
        proposal = [np.floor(0.1 * np.sqrt(chain.shape[0])) for chain in self.chains]
        return proposal

    def plot(self, figsize="COLUMN", filename=None, display=True):
        if isinstance(figsize, str):
            if figsize.upper() == "COLUMN":
                figsize = (5, 5)
            elif figsize.upper() == "PAGE":
                figsize = (10, 10)
            else:
                raise ValueError("Unknown figure size %s" % figsize)
        fig, axes = self.get_figure(figsize=figsize)

        num_bins = self.get_bins()
        fit_values = self.get_summary()
        
        for i, p1 in enumerate(self.all_parameters):
            for j, p2 in enumerate(self.all_parameters):
                ax = axes[i, j]
                if i == j:
                    max_val = None
                    for chain, parameters, colour, bins, fit in zip(self.chains, self.parameters, self.colours, num_bins, fit_values):
                        if p1 not in parameters:
                            continue
                        index = parameters.index(p1)
                        m = self.plot_bars(ax, p1, chain[:, index], colour=colour, bins=bins, fit_values=fit[p1])
                        if max_val is None or m > max_val:
                            max_val = m
                    ax.set_ylim(0, 1.1 * max_val)
                    
                if i > j:
                    for chain, parameters, colour in zip(self.chains, self.parameters, self.colours):
                        if p1 not in parameters or p2 not in parameters:
                            continue
                        i1 = parameters.index(p1)
                        i2 = parameters.index(p2)
                        self.plot_contour(ax, chain[:, i2], chain[:, i1], colour=colour, bins=bins, fit_values=fit)

        if self.names is not None:
            ax = axes[0, -1]
            artists = [plt.Line2D((0,1),(0,0), color=c) for c in self.colours]
            ax.legend(artists, self.names, loc="center", frameon=False)

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300, transparent=True)
        if display:
            plt.show()
        return fig

    def plot_bars(self, ax, parameter, chain_row, bins=50, colour='k', fit_values=None):
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
                ax.set_title(r"$%s = %s$" % (parameter.strip("$"), self.get_parameter_text(*fit_values)), fontsize=14)
        return hist.max()
        
    def clamp(self, val, minimum=0, maximum=255):
        if val < minimum:
            return minimum
        if val > maximum:
            return maximum
        return val
    
    def scale_colours(self, colour, num):
        # http://thadeusb.com/weblog/2010/10/10/python_scale_hex_color
        scales = np.logspace(np.log(0.7), np.log(1.4), num)
        colours = [self.scale_colour(colour, scale) for scale in scales]
        return colours

    def scale_colour(self, colour, scalefactor):
        hex = colour.strip('#')
        if scalefactor < 0 or len(hex) != 6:
            return hex

        r, g, b = int(hex[:2], 16), int(hex[2:4], 16), int(hex[4:], 16)
        r = self.clamp(r * scalefactor)
        g = self.clamp(g * scalefactor)
        b = self.clamp(b * scalefactor)
        return "#%02x%02x%02x" % (r, g, b)
        
    def plot_contour(self, ax, x, y, bins=50, sigmas=None, colour='#222222', fit_values=None):
        if sigmas is None:
            sigmas = np.array([0, 0.5, 1, 2, 3])
        sigmas = np.sort(sigmas)
        levels = 1.0 - np.exp(-0.5 * sigmas ** 2)

        colours = self.scale_colours(colour, len(levels))
        colours2 = [self.scale_colour(c, 0.7) for c in colours]

        hist, x_bins, y_bins = np.histogram2d(x, y, bins=bins)
        x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
        y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
        hist[hist == 0] = 1E-16
        vals = self.convert_to_stdev(hist.T)
        cf = ax.contourf(x_centers, y_centers, vals, levels=levels, colors=colours, alpha=0.8)
        c = ax.contour(x_centers, y_centers, vals, levels=levels, colors=colours2)

    def convert_to_stdev(self, sigma):
        """
        From astroML
    
        Given a grid of log-likelihood values, convert them to cumulative
        standard deviation.  This is useful for drawing contours from a
        grid of likelihoods.
        """
        shape = sigma.shape
        sigma = sigma.ravel()
        i_sort = np.argsort(sigma)[::-1]
        i_unsort = np.argsort(i_sort)
    
        sigma_cumsum = 1.0* sigma[i_sort].cumsum()
        sigma_cumsum /= sigma_cumsum[-1]
    
        return sigma_cumsum[i_unsort].reshape(shape)

    def get_parameter_summary(self, data, parameter, bins=50, desired_area=0.6827):
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
        results = []
        for chain, parameters in zip(self.chains, self.parameters):
            res = {}
            for i, p in enumerate(parameters):
                summary = self.get_parameter_summary(chain[:, i], p)
                res[p] = summary
            results.append(res)
        return results

    def get_parameter_text(self, lower, maximum, upper):
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



                
                
if __name__ == "__main__":
    ndim, nsamples = 3, 100000
    
    # Generate some fake data.
    data1 = np.random.randn(ndim * 4 * nsamples / 5).reshape([4 * nsamples / 5, ndim])
    data2 = (5 * np.random.rand(ndim)[None, :] + np.random.randn(ndim * nsamples / 5).reshape([nsamples / 5, ndim]))
    data = np.vstack([data1, data2])
    
    adata1 = np.random.randn(ndim * 2 * nsamples / 5).reshape([2 * nsamples / 5, ndim])
    adata2 = (1 - 1 * np.random.rand(ndim)[None, :] + np.random.randn(ndim * nsamples / 5).reshape([nsamples / 5, ndim]))
    adata = np.vstack([adata1, adata2])

    labels = ["$x$","$y$",r"$\log\alpha$"]
    
    c = ChainConsumer([data, adata], labels, names=["Chain one", "chain two"])
    c.get_parameter_text(0.00000789, 0.00000801, 0.00000912)
    c.get_parameter_text(0.555553, 0.555555, 0.555559)
    print(c.get_parameter_text(3123210000, 3223210000, 3654310000))
    fig = c.plot()
    fig.savefig("doom.png", bbox_inches="tight", dpi=250)
    #c.get_summary()