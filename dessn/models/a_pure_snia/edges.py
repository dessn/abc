import numpy as np
from astropy.cosmology import FlatwCDM
from dessn.framework.edge import Edge, EdgeTransformation
import sncosmo


class ToLightCurve(Edge):
    def __init__(self):
        super(ToLightCurve, self).__init__("olc", ["mu", "x1", "t0", "c", "abs_mag", "redshift"])
        self.model = sncosmo.Model(source="salt2")

    def get_log_likelihood(self, data):
        r""" Uses SNCosmo to move from supernova parameters to a light curve.
        """
        arr = []
        for z, t0, x1, c, lc, mag, mu in zip(data["redshift"], data["t0"],
                                             data["x1"], data["c"],
                                             data["olc"], data["abs_mag"], data["mu"]):
            self.model.set(z=z)
            self.model.source.set_peakmag(mag + mu, "bessellb", "ab")
            self.model.parameters = [z, t0, self.model.get("x0"), x1, c]
            chi2 = sncosmo.chisq(lc, self.model)
            arr.append(-0.5 * chi2)

        return np.array(arr)


class ToAbsoluteMagnitude(EdgeTransformation):
    def __init__(self):
        super(ToAbsoluteMagnitude, self).__init__("abs_mag", ["magnitude", "alpha",
                                                              "beta", "x1", "c", "delta_m"])

    def get_transformation(self, data):
        val = data["magnitude"] - data["alpha"] * data["x1"] \
              + data["beta"] * data["c"] + data["delta_m"]
        return {"abs_mag": val}


class ToRedshift(EdgeTransformation):
    def __init__(self):
        super(ToRedshift, self).__init__("redshift", ["oredshift"])

    def get_transformation(self, data):
        return {"redshift": data["oredshift"]}


class ToDeltaM(Edge):
    def __init__(self):
        super().__init__("delta_m", "scatter")

    def get_log_likelihood(self, data):
        return -0.5 * data["delta_m"] * data["delta_m"] / (data["scatter"] * data["scatter"])


class ToDistanceModulus(EdgeTransformation):
    def __init__(self):
        super().__init__("mu", ["omega_m", "hubble", "redshift"])

    def get_transformation(self, data):
        cosmology = FlatwCDM(Om0=data["omega_m"], H0=data["hubble"])
        return {"mu": cosmology.distmod(data["redshift"]).value}