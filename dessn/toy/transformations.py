from dessn.model.node import NodeTransformation


class Flux(NodeTransformation):
    def __init__(self):
        super(Flux, self).__init__("Flux", ["flux"], ["$f$"])


class LuminosityDistance(NodeTransformation):
    def __init__(self):
        super(LuminosityDistance, self).__init__("Luminosity Distance", "lumdist", "$d_L$")
