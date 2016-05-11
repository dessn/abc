from dessn.framework.parameter import ParameterLatent, ParameterTransformation
from dessn.models.a_pure_snia.underlying import Scatter
import numpy as np


class Redshift(ParameterTransformation):
    def __init__(self, n):
        super(Redshift, self).__init__("redshift", "$z$", group="Redshift")
        self.n = n

    def get_num_latent(self):
        return self.n


class AbsMag(ParameterTransformation):
    def __init__(self):
        super(AbsMag, self).__init__("abs_mag", "$M_B$", group="Nova. Properties")

#
# class NovaApparentMag(ParameterTransformation):
#     def __init__(self):
#         super(NovaApparentMag, self).__init__("app_mag", "$m_B$", group="Nova. Properties")


# class Scale(ParameterLatent):
#     def __init__(self, n, x0s, x0s_sigma):
#         super().__init__("x0", "$x_0$", group="Salt2")
#         self.n = n
#         self.x0s = x0s
#         self.x0s_sigma = x0s_sigma
#
#     def get_num_latent(self):
#         return self.n
#
#     def get_suggestion_requirements(self):
#         return []
#
#     def get_suggestion(self, data):
#         return self.x0s
#
#     def get_suggestion_sigma(self, data):
#         return 3.0 * self.x0s_sigma


class DeltaMag(ParameterLatent):
    def __init__(self, n):
        super().__init__("delta_m", r"$\Delta M$")
        self.n = n
        self.scatter = Scatter().get_suggestion_sigma({})

    def get_num_latent(self):
        return self.n

    def get_suggestion(self, data):
        return np.zeros(self.n)

    def get_suggestion_sigma(self, data):
        return self.scatter * np.ones(self.n)

    def get_suggestion_requirements(self):
        return []


class Stretch(ParameterLatent):
    def __init__(self, n, x1s, x1s_sigma):
        super().__init__("x1", "$x_1$", group="Salt2")
        self.n = n
        self.x1s = x1s
        self.x1s_sigma = x1s_sigma

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return self.x1s

    def get_suggestion_sigma(self, data):
        return 3.0 * self.x1s_sigma


class PeakTime(ParameterLatent):
    def __init__(self, n, t0s, t0s_sigma):
        super().__init__("t0", "$t_0$", group="Salt2")
        self.n = n
        self.t0s = t0s
        self.t0s_sigma = t0s_sigma

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return self.t0s

    def get_suggestion_sigma(self, data):
        return 3.0 * self.t0s_sigma


class Colour(ParameterLatent):
    def __init__(self, n, cs, cs_sigma):
        super().__init__("c", "$c$", group="Salt2")
        self.n = n
        self.cs = cs
        self.cs_sigma = cs_sigma

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return self.cs

    def get_suggestion_sigma(self, data):
        return 3.0 * self.cs_sigma


class DistanceModulus(ParameterTransformation):
    def __init__(self):
        super().__init__("mu", r"$\mu$", group="Dist. Mod")