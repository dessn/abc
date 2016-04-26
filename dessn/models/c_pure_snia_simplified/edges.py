import numpy as np
from astropy.cosmology import FlatwCDM
from dessn.framework.edge import Edge, EdgeTransformation
import sncosmo


class ToParameters(Edge):
    def __init__(self):
        super(ToParameters, self).__init__(["mb", "x1", "c"], ["mb_o", "x1_o", "c_o", "inv_cov"])

    def get_log_likelihood(self, data):
        m = np.array([data["mb"], data["x1"], data["c"]])
        o = np.array([data["mb_o"], data["x1_o"], data["c_o"]])
        diff = m - o
        icov = data["inv_cov"]
        chi2 = np.dot(diff.T, np.dot(icov, diff))
        return -0.5 * chi2


class ToRedshift(EdgeTransformation):
    def __init__(self):
        super(ToRedshift, self).__init__("redshift", ["oredshift"])

    def get_transformation(self, data):
        return {"redshift": data["oredshift"]}


class ToDistanceModulus(EdgeTransformation):
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
        return {"mu_cos": self.cosmology.distmod(data["redshift"])}


class ToObservedDistanceModulus(EdgeTransformation):
    def __init__(self):
        super().__init__("mu", ["mb", "x1", "c", "alpha", "beta", "mag"])

    def get_transformation(self, data):
        return {"mu": data["mb"] + data["alpha"] * data["x1"] +
                      data["beta"] * data["c"] - data["mag"]}


class ToMus(Edge):
    def __init__(self):
        super().__init__("mu", ["mu_cos", "scatter"])

    def get_log_likelihood(self, data):
        return -0.5 * (data["mu"] - data["mu_cos"])*(data["mu"] - data["mu_cos"]) \
               / (data["scatter"] * data["scatter"])
