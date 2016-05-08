import numpy as np
from astropy.cosmology import FlatwCDM
from dessn.framework.edge import Edge, EdgeTransformation


class ToParameters(Edge):
    r""" The likelihood of true supernova properties given the observed properties.

    Given observed properties :math:`\hat{m}_B, \hat{x}_1, \hat{c}, \hat{C}`, and
    true properties :math:`m_B, x_1, c`, we define :math:`\psi = \left( m_B - \hat{m}_B,
    x_1 - \hat{x}_1, c - \hat{c} \right)`.

    From this, we define the probability

    .. math::
        P(\hat{m}_B, \hat{x}_1, \hat{c}, \hat{C}|m_B,x_1,c) =
        \frac{\exp\left[ -\frac{1}{2} \psi \hat{C}^{-1} \psi^T  \right]}
        {\left| \sqrt{2 \pi \hat{C}} \right| }
    """
    def __init__(self):
        super(ToParameters, self).__init__(["mb", "x1", "c"], ["mb_o", "x1_o", "c_o",
                                                               "inv_cov", "cov"])

    def get_log_likelihood(self, data):
        ls = []
        for mb, x1, c, mb_o, x1_o, c_o, cov, icov in zip(data["mb"], data["x1"], data["c"],
                                                    data["mb_o"], data["x1_o"], data["c_o"],
                                                    data["cov"], data["inv_cov"]):
            o = np.array([mb, x1, c])
            m = np.array([mb_o, x1_o, c_o])
            diff = o - m
            logl = -0.5 * np.dot(diff, np.dot(icov, diff)) \
                   - np.log(np.sqrt(2 * np.pi * np.abs(np.linalg.det(cov))))
            ls.append(logl)
        return np.array(ls)


class ToRedshift(EdgeTransformation):
    r""" Transformation to give true redshift.

    In this simplified model, we assume that the observed redshifts
    have negligible errors and no catastrophic failures, and thus
    the true redshifts are in fact given by the observed redshifts.
    """
    def __init__(self):
        super(ToRedshift, self).__init__("redshift", ["oredshift"])

    def get_transformation(self, data):
        return {"redshift": data["oredshift"]}


class ToDistanceModulus(EdgeTransformation):
    r""" Transformation to give cosmological distance modulus.

    Given :math:`\Omega_m` and :math:`H_0`, we utilise `astropy.cosmology`
    to generate an underlying cosmology. The cosmological distance
    modulus is then calculated from this cosmology and the given redshifts.
    """
    def __init__(self):
        super().__init__("mu_cos", ["omega_m", "H0", "redshift"])
        self.cosmology = None
        self.om = None
        self.H0 = None

    def get_transformation(self, data):
        om = data["omega_m"]
        H0 = data["H0"]
        if not (om == self.om and H0 == self.H0):
            self.cosmology = FlatwCDM(H0=H0, Om0=om)
        return {"mu_cos": self.cosmology.distmod(data["redshift"]).value}


class ToObservedDistanceModulus(EdgeTransformation):
    r""" Transformation to observed distance modulus.

    Given a corrected supernova absolute magnitude of :math:`M = M_B - \alpha x_1 + \beta c`,
    and an observed magnitude of :math:`m_B`, the observed distance modulus is given by:

    .. math::
        \mu_{\rm obs} = m_B - M

        \mu_{\rm obs} = m_B - \alpha x_1 + \beta c - M_B
    """
    def __init__(self):
        super().__init__("mu", ["mb", "x1", "c", "alpha", "beta", "mag"])

    def get_transformation(self, data):
        mus = data["mb"] + data["alpha"] * data["x1"] - data["beta"] * data["c"] - data["mag"]
        return {"mu": mus}


class ToMus(Edge):
    r""" Likelihood of cosmological distance modulus and observed distance modulus.

    Given a cosmological distance modulus :math:`\mu_{\mathcal{C}}` and an observed
    distance modulus :math:`\mu_{\rm obs}`, we define the probability as

    .. math::
        P(\mu_{\rm obs} | \mu_{\mathcal{C}}) = \frac{1}{\sqrt{2\pi}\sigma}
        \exp\left[ -\frac{(\mu_{\rm obs} - \mu_{\mathcal{C}})^2}{2\sigma^2}  \right]

    Our variance here, :math:`\sigma` is given by a combination of the variance
    contributions from the intrinsic scatter of our supernova population :math:`\sigma_{\rm int}`
    and the error in observed distance modulus, which is denoted :math:`\sigma_{\rm obs}` and
    defined via:

    .. math::
        \sigma_{\rm obs}^2 = (1, \alpha, -\beta) \hat{C}^{-1} (1, \alpha, -\beta)^{-1}

    Thus giving the total variance as :math:`\sigma^2 = \sigma_{rm int}^2 + \sigma_{\rm obs}^2`.
    """
    def __init__(self):
        super().__init__("mu", ["mu_cos", "scatter", "alpha", "beta", "cov"])
        self.sqrt2pi = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["mu"] - data["mu_cos"]
        s2 = data["scatter"] * data["scatter"]
        d2s = diff * diff
        sigmas = np.zeros(d2s.shape)
        psiT = np.array([[1, data["alpha"], -data["beta"]]])
        psi = psiT.T
        covs = data["cov"]
        for i, cov in enumerate(covs):
            sigmas[i] = 2 * (s2 + np.abs(np.dot(psiT, np.dot(cov, psi))))
        logl = -d2s / sigmas - self.sqrt2pi - np.log(sigmas)
        return logl
