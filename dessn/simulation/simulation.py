import logging
import numpy as np
from astropy.cosmology import FlatwCDM


class Simulation(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_simulation(self, omega_m=0.3, Zcal=6, H0=70, snIa_luminosity=10, snIa_sigma=0.01, snII_luminosity=9.8,
                       snII_sigma=0.02, sn_rate=0.7, num_trans=50):
        self.logger.info('Getting data for %d transients' % num_trans)
        w_0 = -1

        np.random.seed(1)

        cosmology = FlatwCDM(Om0=omega_m, w0=w_0, H0=H0)

        # Get redshifts with some errors in them
        # z_err_rate = 0.02
        z_err_rate = 0.0
        z = np.exp(np.random.uniform(-3, 3.0, num_trans))
        z2 = np.exp(np.random.uniform(-3, 3.0, num_trans))
        z_err = 1e-5 * np.ones(num_trans)
        z_err_realised = z_err * np.random.normal(0, 1, num_trans)
        catastrophic_failures = 1.0 * (np.random.random(num_trans) < z_err_rate)
        z_o = (z * (1 - catastrophic_failures) + z2 * catastrophic_failures) + z_err_realised

        # From the actual redshift, grab the luminosity distance
        luminosity_distance = cosmology.luminosity_distance(z).value
        # Get the types from the underlying type rate
        type_Ias = 1.0 * (np.random.random(num_trans) < sn_rate)
        type_Iash = ["Ia" if t == 1 else "II" for t in type_Ias]

        misidentification = 1.0 * (np.random.random(num_trans) < 0.1)
        # misidentification = 1.0 * (np.random.random(num_trans) < 0.0)
        type_o = type_Ias * (1 - misidentification) + (1 - type_Ias) * misidentification
        type_oh = ["Ia" if a == 1 else "II" for a in type_o]

        # Get luminosities from type
        luminosity_Ia = snIa_luminosity + np.random.normal(0, snIa_sigma, num_trans)
        luminosity_II = snII_luminosity + np.random.normal(0, snII_sigma, num_trans)
        actual_lum = type_Ias * luminosity_Ia + (1 - type_Ias) * luminosity_II

        # Get flux from luminosity distance and luminosity
        # Remember luminosity is log luminosity
        log_flux = actual_lum - np.log(4 * np.pi * luminosity_distance * luminosity_distance)
        flux = np.exp(log_flux)

        # Get photon counts from flux
        count = flux * np.power(10, Zcal / 2.5)
        count_sigma = np.sqrt(count)
        count_o = count + np.random.normal(0, count_sigma, num_trans)

        observations = {
            "z_o": z_o.tolist(),
            "type_o": type_oh,
            "count_o": count_o.tolist()
        }
        # theta = [omega_m, w_0, h_0, snIa_mean, snIa_sigma, snII_mean, snII_sigma, r] + actual_lum.tolist()
        theta = [omega_m, Zcal, H0, snIa_luminosity, snIa_sigma, snII_luminosity, snII_sigma, sn_rate] + actual_lum.tolist()
        return observations, theta


