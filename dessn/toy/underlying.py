from dessn.model.node import NodeUnderlying


class Cosmology(NodeUnderlying):
    def __init__(self):
        super(Cosmology, self).__init__("Cosmology", ["omega_m", "w"], [r"$\Omega_m$", "$w$"])

    def get_log_prior(self, data):
        return 1


class SupernovaIaDist(NodeUnderlying):
    def __init__(self):
        super(SupernovaIaDist, self).__init__("SNIa", ["snIa_luminosity", "snIa_sigma"], ["$L$", r"$\sigma_L$"])

    def get_log_prior(self, data):
        return 1


class SupernovaIIDist(NodeUnderlying):
    def __init__(self):
        super(SupernovaIIDist, self).__init__("SNII", ["snII_luminosity", "snII_sigma"], ["$L$", r"$\sigma_L$"])

    def get_log_prior(self, data):
        return 1


class SupernovaRate(NodeUnderlying):
    def __init__(self):
        super(SupernovaRate, self).__init__("SN Rates", "sn_rate", "$r$")

    def get_log_prior(self, data):
        return 1




