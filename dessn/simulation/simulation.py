import logging
import numpy as np
from astropy.cosmology import FlatwCDM


class Simulation(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_simulation(self, omega_m, w_0, h_0, snIa_mean, snIa_sigma, snII_mean, snII_sigma, r, num_trans=50):
        self.logger.info('Getting data for %d transients' % num_trans)

        efficiency = 0.9
        conversion = 1e10

        np.random.seed(0)

        cosmology = FlatwCDM(Om0=omega_m, w0=w_0, H0=h_0)

        # Get redshifts with some errors in them
        z_err_rate = 0.02
        z = np.random.uniform(0.1, 2.0, num_trans)
        z2 = np.random.uniform(0.1, 2.0, num_trans)
        z_err = 2e-5 * np.ones(num_trans)
        z_err_realised = z_err * np.random.normal(0, 1, num_trans)
        catastrophic_failures = 1.0 * (np.random.random(num_trans) < z_err_rate)
        z_o = (z * (1 - catastrophic_failures) + z2 * catastrophic_failures) + z_err_realised

        # From the actual redshift, grab the luminosity distance
        luminosity_distance = cosmology.luminosity_distance(z).value

        # Get the types from the underlying type rate
        type_Ias = 1.0 * (np.random.random(num_trans) < r)
        misidentification = 1.0 * (np.random.random(num_trans) < 0.1)
        type_o = type_Ias * (1 - misidentification) + (1 - type_Ias) * misidentification

        # Get luminosities from type
        luminosity_Ia = np.random.normal(snIa_mean, snIa_sigma, num_trans)
        luminosity_II = np.random.normal(snII_mean, snII_sigma, num_trans)
        actual_lum = type_Ias * luminosity_Ia + (1 - type_Ias) * luminosity_II

        # Get flux from luminosity distance and luminosity
        # Remember luminosity is log luminosity
        log_flux = actual_lum - np.log(4 * np.pi) - 2 * np.log(luminosity_distance)
        flux = np.exp(log_flux)

        # Get photon counts from flux
        count = flux * efficiency * conversion
        count_sigma = np.sqrt(count)
        count_o = count + np.random.normal(0, count_sigma, num_trans)

        observations = {
            "z_o": z_o,
            "z_o_err": z_err,
            "type_o": type_o,
            "count_o": count_o
        }
        theta = [omega_m, w_0, h_0, snIa_mean, snIa_sigma, snII_mean, snII_sigma, r] + actual_lum.tolist() + z.tolist() + type_Ias.tolist()
        return observations, theta


