from dessn.model.node import NodeLatent, NodeTransformation, NodeDiscrete


# class Redshift(NodeLatent):
class Redshift(NodeTransformation):
    def __init__(self, n):
        super(Redshift, self).__init__("Redshift", "redshift", "$z$")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return ["oredshift"]

    def get_suggestion(self, data):
        return data["oredshift"].tolist()


class Luminosity(NodeLatent):
    def __init__(self, n):
        super(Luminosity, self).__init__("Luminosity", "luminosity", "$L$")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return ["otype"]

    def get_suggestion(self, data):
        typeIa = data["otype"] == 1.0
        return (typeIa * 10 + (1 - typeIa) * 5.0).tolist()


# class Type(NodeLatent):
class Type(NodeDiscrete):
    def get_types(self):
        return ["Ia", "II"]

    def __init__(self, n):
        super(Type, self).__init__("Type", "type", "$T$")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return ["otype"]

    def get_suggestion(self, data):
        return data["otype"].tolist()
