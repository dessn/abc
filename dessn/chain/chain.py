import matplotlib.pyplot as plt
import numpy as np
import logging
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator


class ChainPlotter(object):
    def __init__(self, chains, parameters):
        self.logger = logging.getLogger(__name__)
        self.all_colours = ["#1E88E5", "#D32F2F", "#4CAF50", "#673AB7", "#FFC107", "#795548", "#64B5F6", "#8BC34A", "#757575", "#CDDC39"]
        self.format_data(chains, parameters)

    def format_data(self, chains, parameters):
        if isinstance(chains, np.ndarray) or isinstance(chains, str):
            logging.info("Single chain and parameters")
            chains = [chains]
            parameters = [parameters]
        else:
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
        proposal = [np.floor(0.15 * np.sqrt(chain.shape[0])) for chain in self.chains]
        return proposal

    def plot(self):
        fig, axes = self.get_figure()

        num_bins = self.get_bins()
        
        for i, p1 in enumerate(self.all_parameters):
            for j, p2 in enumerate(self.all_parameters):
                ax = axes[i,j]
                if i == j:
                    max_val = None
                    for chain, parameters, colour, bins in zip(self.chains, self.parameters, self.colours, num_bins):
                        if p1 not in parameters:
                            continue
                        index = parameters.index(p1)
                        m = self.plot_bars(ax, chain[:, index], colour=colour, bins=bins)
                        if max_val is None or m > max_val:
                            max_val = m
                    ax.set_ylim(0, 1.1 * max_val)
                    
                if i > j:
                    for chain, parameters, colour in zip(self.chains, self.parameters, self.colours):
                        if p1 not in parameters or p2 not in parameters:
                            continue
                        i1 = parameters.index(p1)
                        i2 = parameters.index(p2)
                        self.plot_contour(ax, chain[:, i2], chain[:, i1], colour=colour, bins=bins)

        plt.show()
        return fig

    def plot_bars(self, ax, chain_row, bins=50, colour='k'):
        hist, edges = np.histogram(chain_row, bins=bins, normed=True)
        edge_center = 0.5 * (edges[:-1] + edges[1:])
        ax.hist(edge_center, weights=hist, bins=edges, histtype="step", color=colour)
        
        return hist.max()
        
    def clamp(self, val, minimum=0, maximum=255):
        if val < minimum:
            return minimum
        if val > maximum:
            return maximum
        return val
    
    def scaleColour(self, colour, num):
        # http://thadeusb.com/weblog/2010/10/10/python_scale_hex_color
        scales = np.linspace(0.5, 2, num)
        
        hexstr = colour.strip('#')
        colours = []
        for scalefactor in scales:
            if scalefactor < 0 or len(hexstr) != 6:
                return hexstr
        
            r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)
            r = self.clamp(r * scalefactor)
            g = self.clamp(g * scalefactor)
            b = self.clamp(b * scalefactor)
            colours.append("#%02x%02x%02x" % (r, g, b))
        return colours
        
    def plot_contour(self, ax, x, y, bins=50, levels=None, colour='#222222'):
        if levels is None:
            levels = 1.0 - np.exp(-0.5 * np.arange(0., 2.1, 0.5) ** 2)
            
        colours = self.scaleColour(colour, len(levels))
        L, xBins, yBins = np.histogram2d(x, y, bins=bins)
        L[L == 0] = 1E-16
        vals = self.convert_to_stdev(L.T)
        c = ax.contourf(0.5 * (xBins[:-1] + xBins[1:]), 0.5 * (yBins[:-1] + yBins[1:]), vals, levels=levels, colors=colours, alpha=0.5)
        c = ax.contour(0.5 * (xBins[:-1] + xBins[1:]), 0.5 * (yBins[:-1] + yBins[1:]), vals, levels=levels, colors=colours)

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
                
                
if __name__ == "__main__":
    ndim, nsamples = 3, 100000
    
    # Generate some fake data.
    data1 = np.random.randn(ndim * 4 * nsamples / 5).reshape([4 * nsamples / 5, ndim])
    data2 = (5 * np.random.rand(ndim)[None, :] + np.random.randn(ndim * nsamples / 5).reshape([nsamples / 5, ndim]))
    data = np.vstack([data1, data2])
    
    adata1 = np.random.randn(ndim * 2 * nsamples / 5).reshape([2 * nsamples / 5, ndim])
    adata2 = (1 + 5 * np.random.rand(ndim)[None, :] + np.random.randn(ndim * nsamples / 5).reshape([nsamples / 5, ndim]))
    adata = np.vstack([adata1, adata2])

    labels = ["$x$","$y$",r"$\log\alpha$"]
    
    c = ChainPlotter([data, adata], labels)
    fig = c.plot()
    fig.savefig("doom.png", bbox_inches="tight", dpi=150)
    #c.get_summary()