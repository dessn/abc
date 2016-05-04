import numpy as np
from astropy.cosmology import FlatwCDM
from dessn.framework.edge import Edge, EdgeTransformation


class ToParameters(Edge):
    def __init__(self):
        super(ToParameters, self).__init__(["mb", "x1", "c"], ["mb_o", "x1_o", "c_o", "inv_cov"])

    def get_log_likelihood(self, data):
        ls = []
        for mb, x1, c, mb_o, x1_o, c_o, icov in zip(data["mb"], data["x1"], data["c"],
                                                    data["mb_o"], data["x1_o"], data["c_o"],
                                                    data["inv_cov"]):
            o = np.array([mb, x1, c])
            m = np.array([mb_o, x1_o, c_o])
            diff = o - m
            logl = -0.5 * np.dot(diff, np.dot(icov, diff))
            ls.append(logl)
        return np.array(ls)


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
        return {"mu_cos": self.cosmology.distmod(data["redshift"]).value}


class ToObservedDistanceModulus(EdgeTransformation):
    def __init__(self):
        super().__init__("mu", ["mb", "x1", "c", "alpha", "beta", "mag"])

    def get_transformation(self, data):
        mus = data["mb"] + data["alpha"] * data["x1"] + data["beta"] * data["c"] - data["mag"]
        # print("MUS ", data["mb"], data["alpha"], data["x1"], data["beta"], data["c"], data["mag"], mus)
        return {"mu": mus}


class ToMus(Edge):
    def __init__(self):
        super().__init__("mu", ["mu_cos", "scatter"])
        self.sqrt2pi = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["mu"] - data["mu_cos"]
        s2 = 2 * data["scatter"]*data["scatter"]
        chi2 = diff * diff / s2
        logl = -chi2 - self.sqrt2pi - np.log(data["scatter"])
        return logl
