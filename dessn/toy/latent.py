from dessn.model.node import NodeLatent


class Redshift(NodeLatent):
    def __init__(self, n):
        super(Redshift, self).__init__("Redshift", "redshift", "$z$")
        self.n = n

    def get_num_latent(self):
        return self.n


class Luminosity(NodeLatent):
    def __init__(self, n):
        super(Luminosity, self).__init__("Luminosity", "luminosity", "$L$")
        self.n = n

    def get_num_latent(self):
        return self.n


class Type(NodeLatent):
    def __init__(self, n):
        super(Type, self).__init__("Type", "type", "$T$")
        self.n = n

    def get_num_latent(self):
        return self.n


