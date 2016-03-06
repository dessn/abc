from dessn.model.model import Model
from dessn.toy.edges import ToCount, ToFlux, ToLuminosity, ToLuminosityDistance, ToRate, ToRedshift, ToType
from dessn.toy.latent import Luminosity, Redshift, Type
from dessn.toy.underlying import Cosmology, SupernovaIaDist, SupernovaIIDist, SupernovaRate
from dessn.toy.transformations import Flux, LuminosityDistance
from dessn.toy.observed import ObservedCounts, ObservedRedshift, ObservedType
import logging


class ToyModel(Model):
    """ A modified toy model.


    .. figure::     ../plots/toyModelPGM.png
        :align:     center
    """
    def __init__(self):
        super(ToyModel, self).__init__("ToyModel")

        n = 30

        self.add_node(ObservedType([None]))
        self.add_node(ObservedRedshift([None]))
        self.add_node(ObservedCounts([None]))

        self.add_node(Flux())
        self.add_node(LuminosityDistance())

        self.add_node(Cosmology())
        self.add_node(SupernovaIaDist())
        self.add_node(SupernovaIIDist())
        self.add_node(SupernovaRate())

        self.add_node(Luminosity(n=n))
        self.add_node(Redshift(n=n))
        self.add_node(Type(n=n))

        self.add_edge(ToCount())
        self.add_edge(ToFlux())
        self.add_edge(ToLuminosityDistance())
        self.add_edge(ToLuminosity())
        self.add_edge(ToRedshift())
        self.add_edge(ToRate())
        self.add_edge(ToType())

        self.finalise()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    toy_model = ToyModel()
    fig = toy_model.get_pgm("toyModelPGM.png")
