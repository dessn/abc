import logging

import numpy as np
from astropy.cosmology import FlatwCDM

from observationFactory import ObservationFactory


class Simulation(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_simulation(self, num_trans=30):
        self.logger.info('Getting data for %d transients' % num_trans)

        np.random.seed(0)

        cosmology = FlatwCDM(Om0=0.28, w0=-1., H0=72)

        obs_factory = ObservationFactory(rate_II_r=2.0, cosmology=cosmology, logL_snIa=np.log(1.),
                                         sigma_snIa=0.01, logL_snII=np.log(0.5), sigma_snII=0.4, Z=0.)
        observations = obs_factory.get_observations(num_trans)
        return observations




