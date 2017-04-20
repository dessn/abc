from abc import ABC, abstractmethod
import logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import skewnorm


class Simulation(ABC):

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_approximate_correction(self):
        """ Characterises the simulations selection efficiency in apparent magnitude space as a skew normal. """
        data = self.get_all_supernova(100000)
        print(data["passed"].sum(), data["passed"].size)
        mB_all = data["sim_apparents"]
        mB_passed = mB_all[data["passed"]]

        # Bin data and get ratio
        hist_all, bins = np.histogram(mB_all, bins=100)
        hist_passed, _ = np.histogram(mB_passed, bins=bins)
        binc = 0.5 * (bins[:-1] + bins[1:])
        hist_all[hist_all == 0] = 1
        ratio = hist_passed / hist_all

        # Inverse transformation sampling to sample from this random pdf
        cdf = ratio.cumsum()
        cdf = cdf / cdf.max()
        cdf[0] = 0
        cdf[-1] = 1
        n = 100000
        u = np.random.random(size=n)
        y = interp1d(cdf, binc)(u)

        alpha, mean, std = skewnorm.fit(y)
        self.logger.info("Fitted efficiency to have mean %0.2f, std %0.2f and alpha %0.2f" % (mean, std, alpha))

        # import matplotlib.pyplot as plt
        # print(mB.shape)
        # plt.plot(binc, ratio * skewnorm.pdf(mean, alpha, mean, std))
        # plt.plot(binc, skewnorm.pdf(binc, alpha, mean, std))
        # plt.hist(y, 100, histtype='step', normed=True)
        # plt.show()
        # exit()

        return mean, std, alpha

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
