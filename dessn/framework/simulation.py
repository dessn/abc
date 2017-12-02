from abc import ABC, abstractmethod
import logging
import numpy as np


class Simulation(ABC):

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def get_approximate_correction(self):
        raise NotImplementedError()

    def get_passed_supernova(self, n_sne, cosmology_index=0):
        result = self.get_all_supernova(n_sne, cosmology_index=cosmology_index)
        mask = result["passed"]
        for k in list(result.keys()):
            if isinstance(result[k], np.ndarray):
                result[k] = result[k][mask]
        del result["passed"]
        return result

    def get_systematic_names(self):
        return []

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
