import numpy as np
from astropy.cosmology import FlatwCDM
from scipy import special
from dessn.model.edge import Edge, EdgeTransformation, EdgeDiscrete


class ToCount(Edge):

    def __init__(self):
        super(ToCount, self).__init__("ocount", "flux")

    def get_log_likelihood(self, data):
        r""" Given CCD efficiency, convert from count to flux :math:`f`. We go from counts, assume Poisson error, to get
        an observed flux and flux error from counts. From that, we can calculate the likelihood of an actual flux, given
        our observed flux and observed flux error, assuming a normal distribution:

        .. math::
            f_{\rm observed} = \frac{{\rm count}}{{\rm conversion} \times {\rm efficiency}}

            \sigma_f = \frac{\sqrt{{\rm count}}}{{\rm conversion} \times {\rm efficiency}}

            P ({\rm count}_o | f_ = \frac{1}{\sqrt{2\pi}} \exp\left( - \frac{(f_{\rm observed}-f)^2}{2 \sigma_f^2} \right)

        """
        efficiency = 0.9
        conversion = 1e10
        flux = data["flux"]
        f = data["ocount"] / efficiency / conversion
        fe = np.sqrt(data["ocount"]) / efficiency / conversion

        return np.sum(-(flux - f) * (flux - f) / (2 * fe * fe) - np.log(np.sqrt(2 * np.pi) * fe))


class ToFlux(EdgeTransformation):
    def __init__(self):
        super(ToFlux, self).__init__("flux", ["lumdist", "luminosity"])

    def get_transformation(self, data):
        r""" Gets flux from the luminosity distance and luminosity. Note that the luminosity here is
        actually the **log luminosity**

        .. math::
            f = \frac{L}{4\pi D_L^2}

        """
        flux = np.exp(data["luminosity"]) / (4 * np.pi * data["lumdist"] * data["lumdist"])
        return {"flux": flux}


class ToLuminosityDistance(EdgeTransformation):
    def __init__(self):
        super(ToLuminosityDistance, self).__init__("lumdist", ["omega_m", "w", "H0", "redshift"])

    def get_transformation(self, data):
        if data["omega_m"] < 0:
            return -np.inf
        cosmology = FlatwCDM(H0=data["H0"], Om0=data["omega_m"], w0=data["w"])
        return {"lumdist": cosmology.luminosity_distance(data['redshift']).value}


# class ToRedshift(Edge):
class ToRedshift(EdgeTransformation):
    def get_transformation(self, data):
        return {"redshift": data["oredshift"]}

    def __init__(self):
        super(ToRedshift, self).__init__("redshift", ["oredshift", "oredshift_error"])

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

        # Note that we are working with arrays for type and luminosity, one element per supernova
        snIa_mask = (sn_type == 1)
        snII_mask = (sn_type == 0)

        luminosity = data["luminosity"]
        snIa_mean = data["snIa_luminosity"]
        snIa_std = data["snIa_sigma"]
        snII_mean = data["snII_luminosity"]
        snII_std = data["snII_sigma"]

        snIa_prob = (-(luminosity - snIa_mean) * (luminosity - snIa_mean) / (2 * snIa_std * snIa_std)) - np.log(np.sqrt(2 * np.pi) * snIa_std)
        snII_prob = (-(luminosity - snII_mean) * (luminosity - snII_mean) / (2 * snII_std * snII_std)) - np.log(np.sqrt(2 * np.pi) * snII_std)
        # print(snIa_prob, snII_prob)
        return np.sum(snIa_mask * snIa_prob + snII_mask * snII_prob)


# class ToType(Edge):
class ToType(EdgeDiscrete):
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
            P(T_o|T) = 0.1 + 0.8\delta_{T_o,T}
        """

        o_type = data["otype"]
        input_type = data["type"]
        prob = 0.9 * (o_type == input_type) + 0.1

        return np.log(prob)


class ToRate(Edge):
    def __init__(self):
        super(ToRate, self).__init__("type", "sn_rate")

    def get_log_likelihood(self, data):
        r""" The likelihood of having the supernova types :math:`T` given supernova rate :math:`r`.

        We model the supernova rate as a binomial process, with rate :math:`r`. That is, given :math:`x` type
        Ia supernova and :math:`y` type II supernova, our pdf is given by

        .. math::
            P(T|r) = \begin{pmatrix} N_{\rm Total} \\ N_{\rm SnII} \end{pmatrix} r^{N_{\rm SnIa}} (1 - r)^{N_{\rm SnII}}

        In log space, this is

        .. math::
            \log(P(T|r)) = \log \begin{pmatrix} N_{\rm Total} \\ N_{\rm SnII} \end{pmatrix} + N_{\rm SnIa} \log(r) + N_{\rm SnII} \log(1-r)

        In the code, I approximate the choose function using the log gamma functions.
        """
        if data["sn_rate"] < 0 or data["sn_rate"] > 1:
            return -np.inf

        sn_type = data["type"]
        n_snIa = (sn_type == 1.0).sum()
        n_snII = (sn_type != 1.0).sum()
        n = sn_type.size
        r = data["sn_rate"]

        # TODO: Probably want generic classes for Normals, log normals, binomial, etc
        log_choose = special.gammaln(n + 1) - special.gammaln(n_snII + 1) - special.gammaln(n_snIa + 1)
        # print(log_choose, n_snIa, n_snII, r, n, data["sn_rate"], data["type"])
        return log_choose + n_snIa * np.log(r) + n_snII * np.log(1-r)
