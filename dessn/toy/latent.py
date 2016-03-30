from dessn.model.parameter import ParameterLatent, ParameterTransformation, ParameterDiscrete


class Redshift(ParameterTransformation):
    def __init__(self, n):
        super(Redshift, self).__init__("redshift", "$z$", group="Redshift")
        self.n = n

    def get_num_latent(self):
        return self.n


class Luminosity(ParameterLatent):
    def __init__(self, n):
        super(Luminosity, self).__init__("luminosity", "$L$", group="Luminosity")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return ["otype"]

    def get_suggestion(self, data):
        typeIa = data["otype"] == "Ia"
        return typeIa * 10 + (1 - typeIa) * 9.8

    def get_suggestion_sigma(self, data):
        return 0.1


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
