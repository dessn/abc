from dessn.framework.parameter import ParameterLatent, ParameterTransformation
import numpy as np


class Redshift(ParameterTransformation):
    def __init__(self, n):
        super(Redshift, self).__init__("redshift", "$z$", group="Redshift")
        self.n = n

    def get_num_latent(self):
        return self.n


class Luminosity(ParameterLatent):
    def __init__(self, n):
        super(Luminosity, self).__init__("luminosity", "$L$", group="Supernova Properties")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion(self, data):
        return -19.3

    def get_suggestion_sigma(self, data):
        return 0.001

    def get_suggestion_requirements(self):
        return []


class Stretch(ParameterLatent):
    def __init__(self, n):
        super().__init__("x1", "$x_1$", group="Supernova Properties")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 0

    def get_suggestion_sigma(self, data):
        return 2.0


class PeakTime(ParameterLatent):
    def __init__(self, n):
        super().__init__("t0", "$t_0$", group="Supernova Properties")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return ["olc"]

    def get_suggestion(self, data):
        return np.mean(data["olc"]["time"])

    def get_suggestion_sigma(self, data):
        return 30


class Colour(ParameterLatent):
    def __init__(self, n):
        super().__init__("c", "$c$", group="Supernova Properties")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 0

    def get_suggestion_sigma(self, data):
        return 0.3
