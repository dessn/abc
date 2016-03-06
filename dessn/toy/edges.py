from dessn.model.edge import Edge, EdgeTransformation


class ToCount(EdgeTransformation):
    def __init__(self):
        super(ToCount, self).__init__("ocount", "flux")

    def get_transformation(self, data):
        pass


class ToFlux(EdgeTransformation):
    def __init__(self):
        super(ToFlux, self).__init__("flux", ["lumdist", "luminosity"])

    def get_transformation(self, data):
        pass


class ToLuminosityDistance(EdgeTransformation):
    def __init__(self):
        super(ToLuminosityDistance, self).__init__("lumdist", ["omega_m", "w", "redshift"])

    def get_transformation(self, data):
        pass


class ToRedshift(Edge):
    def __init__(self):
        super(ToRedshift, self).__init__("oredshift", "redshift")

    def get_log_likelihood(self, data):
        pass


class ToLuminosity(Edge):
    def __init__(self):
        super(ToLuminosity, self).__init__("luminosity", ["type", "snIa_luminosity", "snIa_sigma", "snII_luminosity", "snII_sigma"])

    def get_log_likelihood(self, data):
        pass


class ToType(Edge):
    def __init__(self):
        super(ToType, self).__init__("otype", "type")

    def get_log_likelihood(self, data):
        pass


class ToRate(Edge):
    def __init__(self):
        super(ToRate, self).__init__("type", "sn_rate")

    def get_log_likelihood(self, data):
        pass
