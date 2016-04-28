from dessn.framework.parameter import ParameterLatent, ParameterTransformation
import numpy as np


class Redshift(ParameterTransformation):
    def __init__(self, n):
        super(Redshift, self).__init__("redshift", "$z$", group="Redshift")
        self.n = n


class CosmologicalDistanceModulus(ParameterTransformation):
    def __init__(self):
        super().__init__("mu_cos", r"$\mu$")


class ObservedDistanceModulus(ParameterTransformation):
    def __init__(self):
        super().__init__("mu", r"$\mu_B$")


class ApparentMagnitude(ParameterLatent):
    def __init__(self, n):
        super(ApparentMagnitude, self).__init__("mb", "$m_B$", group="Supernova Properties")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion(self, data):
        return data["mb_o"]

    def get_suggestion_sigma(self, data):
        return 2 / data["inv_cov"][0, 0]

    def get_suggestion_requirements(self):
        return ["mb_o", "inv_cov"]


class Stretch(ParameterLatent):
    def __init__(self, n):
        super().__init__("x1", "$x_1$", group="Supernova Properties")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return ["inv_cov", "x1_o"]

    def get_suggestion(self, data):
        return data["x1_o"]

    def get_suggestion_sigma(self, data):
        return 2 / data["inv_cov"][1, 1]


class Colour(ParameterLatent):
    def __init__(self, n):
        super().__init__("c", "$c$", group="Supernova Properties")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return ["inv_cov", "c_o"]

    def get_suggestion(self, data):
        return data["c_o"]

    def get_suggestion_sigma(self, data):
        return 2.0 / data["inv_cov"][2, 2]

