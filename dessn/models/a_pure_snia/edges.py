import numpy as np
from astropy.cosmology import FlatwCDM
from dessn.framework.edge import Edge, EdgeTransformation
import sncosmo


class ToLightCurve(Edge):
    def __init__(self):
        super(ToLightCurve, self).__init__("olc", ["redshift", "x0", "x1", "t0", "c"])
        self.model = sncosmo.Model(source="salt2")

    def get_log_likelihood(self, data):
        r""" Uses SNCosmo to move from supernova parameters to a light curve.
        """
        arr = []
        for z, t0, x0, x1, c, lc in zip(data["redshift"], data["t0"],
                                    data["x0"], data["x1"], data["c"], data["olc"]):
            self.model.set(z=z)
            self.model.parameters = [z, t0, x0, x1, c]
            chi2 = sncosmo.chisq(lc, self.model)
            arr.append(-0.5 * chi2)

        return np.array(arr)


class ToApparentMagnitude(EdgeTransformation):
    def __init__(self):
        super(ToApparentMagnitude, self).__init__("app_mag", ["x0", "x1", "c"])
        self.model = sncosmo.Model(source="salt2")

    def get_transformation(self, data):
        arr = []
        for x0, x1, c in zip(data["x0"], data["x1"], data["c"]):
            self.model.set(x0=x0, x1=x1, c=c)
            app = self.model.source.peakmag("bessellb", "ab")
            arr.append(app)
        return {"app_mag": np.array(arr)}


class ToAbsoluteMagnitude(EdgeTransformation):
    def __init__(self):
        super(ToAbsoluteMagnitude, self).__init__("abs_mag",
                                                  ["magnitude", "alpha", "beta", "x1", "c"])

    def get_transformation(self, data):
        val = data["magnitude"] - data["alpha"] * data["x1"] + data["beta"] * data["c"]
        return {"abs_mag": val}


class ToObservedDistanceModulus(EdgeTransformation):
    def __init__(self):
        super().__init__("mu_obs", ["abs_mag", "app_mag"])

    def get_transformation(self, data):
        return {"mu_obs": data["app_mag"] - data["abs_mag"]}


class ToCosmologicalDistanceModulus(EdgeTransformation):
    def __init__(self):
        super().__init__("mu_cos", ["omega_m", "hubble", "redshift"])

    def get_transformation(self, data):
        cosmology = FlatwCDM(H0=data["hubble"], Om0=data["omega_m"])
        return {"mu_cos": cosmology.distmod(data["redshift"]).value}


class ToDistanceModuli(Edge):
    def __init__(self):
        super().__init__("mu_obs", ["mu_cos", "scatter"])

    def get_log_likelihood(self, data):
        diff = data["mu_obs"] - data["mu_cos"]
        return -0.5 * diff * diff / (data["scatter"] * data["scatter"])


class ToRedshift(EdgeTransformation):
    def __init__(self):
        super(ToRedshift, self).__init__("redshift", ["oredshift"])

    def get_transformation(self, data):
        return {"redshift": data["oredshift"]}

