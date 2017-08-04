from abc import ABC, abstractmethod
import logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import skewnorm, norm
from scipy.ndimage.filters import gaussian_filter1d


class Simulation(ABC):

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_approximate_correction(self, plot=False):
        """ Characterises the simulations selection efficiency in apparent magnitude space as a skew normal. """
        data = self.get_all_supernova(100000)
        self.logger.info("Got data to compute selection function")
        mB_all = data["sim_apparents"]
        mB_passed = mB_all[data["passed"]]

        # Bin data and get ratio
        hist_all, bins = np.histogram(mB_all, bins=100)
        hist_passed, _ = np.histogram(mB_passed, bins=bins)
        binc = 0.5 * (bins[:-1] + bins[1:])
        hist_all[hist_all == 0] = 1
        ratio = hist_passed / hist_all
        ratio_smooth = gaussian_filter1d(ratio, 2)

        is_cdf = ratio[:5].mean() > 0.8
        if not is_cdf:
            # Inverse transformation sampling to sample from this random pdf

            cdf = ratio_smooth.cumsum()
            cdf = cdf / cdf.max()
            cdf[0] = 0
            cdf[-1] = 1
            n = 100000
            u = np.random.random(size=n)
            y = interp1d(cdf, binc)(u)

            alpha, mean, std = skewnorm.fit(y)

            if mean < 15:
                mean -= 0.3

            if np.abs(alpha) > 10:
                is_cdf = True
            else:
                normm = ratio.max()
                self.logger.info("Fitted skewnorm efficiency to have mean %0.2f, std %0.2f and alpha %0.2f" % (mean, std, alpha))

                if plot:
                    import matplotlib.pyplot as plt
                    plt.plot(binc, 1.0 * hist_passed / hist_passed.max(), label="Passed")
                    plt.plot(binc, ratio, label="Ratio")
                    plt.plot(binc, ratio_smooth, label="Ratio smoothed")
                    plt.plot(binc, skewnorm.pdf(binc, alpha, mean, std), label="PDF")
                    plt.hist(y, 100, histtype='step', normed=True, label="Sampled Hist")
                    plt.legend()
                    plt.show()
                    exit()

        if is_cdf:
            ratio_smooth[:ratio_smooth.argmax()] = ratio_smooth.max()
            ratio_smooth /= ratio_smooth.max()
            vals = [0.5 - 0.68/2, 0.5, 0.5 + 0.68/2]
            mags = interp1d(ratio_smooth, binc)(vals)
            mean = mags[1]
            std = 0.51 * np.abs(mags[0] - mags[2])  # 0.51 not 0.5 because better to overestimate than under
            alpha, normm = None, 1.0
            self.logger.info("Fitted cdf efficiency to have mean %0.2f, std %0.2f" % (mean, std))

            if plot:
                import matplotlib.pyplot as plt
                plt.plot(binc, 1.0 * hist_passed / hist_passed.max(), label="Passed")
                plt.plot(binc, ratio, label="Ratio")
                plt.plot(binc, ratio_smooth, label="Ratio smoothed")
                plt.plot(binc, 1 - norm.cdf(binc, mean, std), label="PDF")
                plt.legend()
                plt.show()
                exit()

        return mean, std, alpha, normm

    def get_passed_supernova(self, n_sne, simulation=True, cosmology_index=0):
        result = self.get_all_supernova(n_sne, cosmology_index=cosmology_index)
        mask = result["passed"]
        for k in list(result.keys()):
            if isinstance(result[k], np.ndarray):
                result[k] = result[k][mask]
        del result["passed"]
        return result

    def get_truth_values_dict(self):
        vals = self.get_truth_values()
        return {k[0]: k[1] for k in vals}

    @abstractmethod
    def get_name(self):
        raise NotImplementedError()

    @abstractmethod
    def get_truth_values(self):
        raise NotImplementedError()

    @abstractmethod
    def get_all_supernova(self, n_sne, cosmology_index=0):
        raise NotImplementedError()
