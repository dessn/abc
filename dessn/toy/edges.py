import numpy as np
from astropy.cosmology import FlatwCDM
from dessn.model.edge import Edge, EdgeTransformation


class ToCount(Edge):

    def __init__(self):
        super(ToCount, self).__init__("ocount", "flux")
        self.sqrt2pi = np.sqrt(2 * np.pi)

    def get_log_likelihood(self, data):
        r""" Given CCD efficiency, convert from count to flux :math:`f`. We go from counts, assume Poisson error, to get
        an observed flux and flux error from counts. From that, we can calculate the likelihood of an actual flux, given
        our observed flux and observed flux error, assuming a normal distribution:

        .. math::
            f_{\rm observed} = \frac{{\rm count}}{{\rm conversion} \times {\rm efficiency}}

            \sigma_f = \frac{\sqrt{{\rm count}}}{{\rm conversion} \times {\rm efficiency}}

            P ({\rm count}_o | f) = \frac{1}{\sqrt{2\pi} \sigma_f} \exp\left( - \frac{(f_{\rm observed}-f)^2}{2 \sigma_f^2} \right)

        """
        efficiency = 0.9
        conversion = 1e7
        flux = data["flux"]
        f = data["ocount"] / efficiency / conversion
        fe = np.sqrt(data["ocount"]) / efficiency / conversion

        return -(flux - f) * (flux - f) / (2 * fe * fe) - np.log(self.sqrt2pi * fe)


class ToFlux(EdgeTransformation):
    def __init__(self):
        super(ToFlux, self).__init__("flux", ["lumdist", "luminosity"])

    def get_transformation(self, data):
        r""" Gets flux from the luminosity distance and luminosity. Note that the luminosity here is
        actually the **log luminosity**

        .. math::
            f = \frac{L}{4\pi D_L^2}

        """
        flux = np.exp(data["luminosity"] - np.log(4 * np.pi * data["lumdist"] * data["lumdist"]))
        return {"flux": flux}


class ToLuminosityDistance(EdgeTransformation):
    def __init__(self):
        super(ToLuminosityDistance, self).__init__("lumdist", ["omega_m", "H0", "redshift"]) # "w", "H0",

    def get_transformation(self, data):
        if data["omega_m"] < 0:
            return {"lumdist": -np.inf}
        cosmology = FlatwCDM(H0=data["H0"], Om0=data["omega_m"], w0=-1)
        return {"lumdist": cosmology.luminosity_distance(data['redshift']).value}


# class ToRedshift(Edge):
class ToRedshift(EdgeTransformation):
    def get_transformation(self, data):
        return {"redshift": data["oredshift"]}

    def __init__(self):
        super(ToRedshift, self).__init__("redshift", ["oredshift"])

        '''
        super(ToRedshift, self).__init__(["oredshift", "oredshift_error"], "redshift")
    def get_log_likelihood(self, data):
        r""" Assume the redshift distribution follows a uniform distribution (for misidentification)
        with a tight Gaussian peak around the observed redshift.

        Assumes the misidentifiation range is between :math:`z=0` and :math:`z=2`.
        Also assumes the success rate is 99% for observed spectra

        .. math::
            P(z_o|z) = \frac{0.01}{2} + \frac{0.99}{\sqrt{2\pi} z_{o,{\rm err}}} \exp\left(  -\frac{(z-z_o)^2}{2z^2_{o,{\rm err}}}  \right)

        """
        uniform = np.log(0.01 / 2)
        gauss = -(data["oredshift"] - data["redshift"]) * (data["oredshift"] - data["redshift"]) / (2 * data["oredshift_error"] * data["oredshift_error"])
        gauss -= np.log(np.sqrt(2 * np.pi) * data["oredshift_error"])
        result = np.logaddexp(gauss, uniform)
        return np.sum(result)'''


class ToLuminosity(Edge):
    def __init__(self):
        super(ToLuminosity, self).__init__("luminosity", ["type", "snIa_luminosity", "snIa_sigma", "snII_luminosity", "snII_sigma"])
        self.sqrt2pi = np.sqrt(2 * np.pi)

    def get_log_likelihood(self, data):
        r""" Assume type is 0 for a Type SnIa, or 1 for SnII. It will be continuous, so we round the variable.

        If we have a type SnIa supernova, we use the type SnIa distribution, which is modelled as a gaussian.

        We should also note clearly that luminosity here is actually **log luminosity**, we work in log space.

        .. math::
            P(L|\mu_{\rm SnIa}, \sigma_{\rm SnIa}) = \frac{1}{\sqrt{2\pi}\sigma_{\rm SnIa}} \exp\left( - \frac{(L - \mu_{\rm SnIa})^2}{2\sigma_{\rm SnIa}} \right)

        If we have a type SnII supernova, we use the type SnII distribution, which is also modelled as a gaussian.

        .. math::
            P(L|\mu_{\rm SnII}, \sigma_{\rm SnII}) = \frac{1}{\sqrt{2\pi}\sigma_{\rm SnII}} \exp\left( - \frac{(L - \mu_{\rm SnII})^2}{2\sigma_{\rm SnII}} \right)

        """

        # TODO: Where should we consistency check parameters? Should we need to?
        sn_type = data["type"]
        luminosity = data["luminosity"]
        if sn_type == "Ia":
            snIa_mean = data["snIa_luminosity"]
            snIa_std = data["snIa_sigma"]
            snIa_prob = (-(luminosity - snIa_mean) * (luminosity - snIa_mean) / (2 * snIa_std * snIa_std)) - np.log(self.sqrt2pi * snIa_std)
            return snIa_prob
        elif sn_type == "II":
            snII_mean = data["snII_luminosity"]
            snII_std = data["snII_sigma"]
            snII_prob = (-(luminosity - snII_mean) * (luminosity - snII_mean) / (2 * snII_std * snII_std)) - np.log(self.sqrt2pi * snII_std)
            return snII_prob
        else:
            raise ValueError("Unrecognised type: %s" % sn_type)
            return -np.inf


# class ToType(Edge):
class ToType(Edge):
    def __init__(self):
        super(ToType, self).__init__("otype", "type")

    def get_log_likelihood(self, data):
        r""" Gets the probability of the actual object being of one type, given we observe a singular other type.

        That is, if we think we observe a type Ia supernova, what is the probability it is actually a type Ia, and
        what is the probability it is a different type of supernova.

        At the moment, this is a trivial function, where we assume that we are correct 90% of the time.

        Also note that the input types (accessed by the ``type`` key) are continuous, and we therefore round them
        to get the discrete type. The method of changing from continuous to discrete will probably update in the future.

        .. math::
            P(T_o|T) = 0.01 + 0.98\delta_{T_o,T}
        """

        o_type = data["otype"]
        input_type = data["type"]
        prob = 0.98 * (o_type == input_type) + 0.01

        return np.log(prob)


class ToRate(Edge):
    def __init__(self):
        super(ToRate, self).__init__("type", "sn_rate")

    def get_log_likelihood(self, data):
        r""" The likelihood of having the supernova types :math:`T` given supernova rate :math:`r`.

        We model the supernova rate as a binomial process, with rate :math:`r`. That is, given :math:`x` type
        Ia supernova and :math:`y` type II supernova, our pdf is given by

        .. math::
            P(T|r) = \begin{cases}
                r
                ,& \text{if } T = {\rm SnIa} \\
                1-r,
                & \text{if } T = {\rm SnII}
                \end{cases}

        """
        r = data["sn_rate"]
        if r < 0 or r > 1:
            return -np.inf

        sn_type = data["type"]
        if sn_type == "Ia":
            return np.log(r)
        elif sn_type == "II":
            return np.log(1 - r)
        else:
            return -np.inf

