from dessn.model.parameter import ParameterLatent, ParameterTransformation, ParameterDiscrete
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

    def get_suggestion_requirements(self):
        return ["otype"]

    def get_suggestion(self, data):
        typeIa = data["otype"] == "Ia"
        return typeIa * -19.3 + (1 - typeIa) * -18.5

    def get_suggestion_sigma(self, data):
        return 0.2


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
        return 1.0


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


class Type(ParameterDiscrete):
    def get_discrete_requirements(self):
        return []

    def get_discrete(self, data):
        return ["Ia", "II"]

    def __init__(self, n):
        super(Type, self).__init__("type", "$T$", group="Type")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return ["otype"]

    def get_suggestion(self, data):
        return data["otype"]
