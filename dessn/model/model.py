import numpy as np
import logging
import os
from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, filename):
        self.logger = logging.getLogger(__name__)
        self.filename = filename
        self.name = os.path.basename(filename)

    @abstractmethod
    def get_init(self):
        raise NotImplementedError()

    @abstractmethod
    def get_data(self, simulation, cosmology_index):
        raise NotImplementedError()

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError()

    def correct_chain(self, dictionary, simulation):
        return dictionary