import numpy as np
from astropy.cosmology import FlatwCDM
from dessn.framework.edge import Edge, EdgeTransformation
import sncosmo


class ToLightCurve(Edge):
    def __init__(self):
        super(ToLightCurve, self).__init__("olc", ["redshift", "luminosity", "x1",
                                                   "t0", "c", "omega_m", "H0"])
        self.model = sncosmo.Model(source="salt2")
        self.current_h0 = None
        self.current_omega_m = None
        self.cosmology = None

    def get_log_likelihood(self, data):
        r""" Uses SNCosmo to move from supernova parameters to a light curve.
        """
        H0 = data["H0"]
        om = data["omega_m"]
        if self.cosmology is None or (self.current_h0 != H0 or self.current_omega_m != om):
            self.cosmology = FlatwCDM(H0=H0, Om0=om, w0=-1)
            self.current_h0 = H0
            self.current_omega_m = om
        self.model.set(z=data["redshift"])
        self.model.set_source_peakabsmag(data["luminosity"], 'bessellb', 'ab', cosmo=self.cosmology)
        x0 = self.model.get("x0")
        self.model.parameters = [data["redshift"], data["t0"], x0, data["x1"], data["c"]]

        chi2 = sncosmo.chisq(data["olc"], self.model)
        return -0.5 * chi2


class ToRedshift(EdgeTransformation):
    def __init__(self):
        super(ToRedshift, self).__init__("redshift", ["oredshift"])

    def get_transformation(self, data):
        return {"redshift": data["oredshift"]}


class ToLuminosity(Edge):
    def __init__(self):
        super(ToLuminosity, self).__init__("luminosity", ["snIa_luminosity", "snIa_sigma"])
        self.sqrt2pi = np.sqrt(2 * np.pi)

    def get_log_likelihood(self, data):
        r""" Assume type is "Ia" for a Type SnIa, or "II" for SnII.

        If we have a type SnIa supernova, we use the type SnIa distribution, which is
        modelled as a gaussian.

        We should also note clearly that luminosity here is actually peak absolute magnitude

        .. math::
            P(L|\mu_{\rm SnIa}, \sigma_{\rm SnIa}) = \frac{1}{\sqrt{2\pi}\sigma_{\rm SnIa}}
            \exp\left( - \frac{(L - \mu_{\rm SnIa})^2}{2\sigma_{\rm SnIa}} \right)

        If we have a type SnII supernova, we use the type SnII distribution, which is also
        modelled as a gaussian.

        .. math::
            P(L|\mu_{\rm SnII}, \sigma_{\rm SnII}) = \frac{1}{\sqrt{2\pi}\sigma_{\rm SnII}}
            \exp\left( - \frac{(L - \mu_{\rm SnII})^2}{2\sigma_{\rm SnII}} \right)

        """

        luminosity = data["luminosity"]
        snIa_mean = data["snIa_luminosity"]
        snIa_std = data["snIa_sigma"]
        snIa_prob = (-(luminosity - snIa_mean) * (luminosity - snIa_mean) /
                     (2 * snIa_std * snIa_std)) - np.log(self.sqrt2pi * snIa_std)
        return snIa_prob


