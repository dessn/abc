import numpy as np
from astropy.cosmology import FlatwCDM

from dessn.model.edge import Edge, EdgeTransformation
from dessn.utility.math import plus


class ToCount(EdgeTransformation):
    def __init__(self):
        super(ToCount, self).__init__("ocount", ["flux", "flux_error"])

    def get_transformation(self, data):
        r""" Given CCD efficiency, convert from count to flux :math:`f` and flux error :math:`\sigma_f`.

        .. math::
            f = \frac{{\rm count}}{{\rm conversion} \times {\rm efficiency}}

            \sigma_f = \sqrt{f}
        """
        efficiency = 0.9
        conversion = 1000
        flux = (data["ocount"] / efficiency / conversion)
        flux_error = np.sqrt(flux)
        return {"flux": flux, "flux_error": flux_error}


class ToFlux(EdgeTransformation):
    def __init__(self):
        super(ToFlux, self).__init__("flux", ["lumdist", "luminosity"])

    def get_transformation(self, data):
        r""" Gets flux from the luminosity distance and luminosity.

        .. math::
            f = \frac{L}{4\pi D_L^2}

        """
        flux = data["luminosity"] / (4 * np.pi * data["lumdist"] * data["lumdist"])
        return {"flux": flux}


class ToLuminosityDistance(EdgeTransformation):
    def __init__(self):
        super(ToLuminosityDistance, self).__init__("lumdist", ["omega_m", "w", "H0", "redshift"])

    def get_transformation(self, data):
        cosmology = FlatwCDM(H0=data["H0"], Om0=data["omega_m"], w0=data["w"])
        return {"lumdist": cosmology.luminosity_distance(data['redshift']).value}


class ToRedshift(Edge):
    def __init__(self):
        super(ToRedshift, self).__init__(["oredshift", "oredshift_error"], "redshift")

    def get_log_likelihood(self, data):
        r""" Assume the redshift distribution follows a uniform distribution (for misidentification)
        with a tight Gaussian peak around the observed redshift.

        Assumes the misidentifiation range is between :math:`z=0` and :math:`z=2`.
        Also assumes the success rate is 99% for observed spectra

        .. math::
            P(z_o|z) = \frac{0.01}{2} + \frac{0.99}{\sqrt{2\pi} z_{o,{\rm err}}} \exp\left(  -\frac{(z-z_o)^2}{2z^2_{o,{\rm err}}}  \right)

        """
        uniform = np.log(0.01/2)
        gauss = -(data["oredshift"] - data["redshift"]) * (data["oredshift"] - data["redshift"]) / (2 * data["oredshift_error"] * data["oredshift_error"])\
                - np.log(np.sqrt(2 * np.pi) * data["oredshift_error"])
        return plus(uniform, gauss)


class ToLuminosity(Edge):
    def __init__(self):
        super(ToLuminosity, self).__init__("luminosity", ["type", "snIa_luminosity", "snIa_sigma", "snII_luminosity", "snII_sigma"])

    def get_log_likelihood(self, data):
        pass


class ToType(Edge):
    def __init__(self):
        super(ToType, self).__init__("otype", "type")

    def get_log_likelihood(self, data):
        pass


class ToRate(Edge):
    def __init__(self):
        super(ToRate, self).__init__("type", "sn_rate")

    def get_log_likelihood(self, data):
        pass
