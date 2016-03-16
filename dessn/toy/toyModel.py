from dessn.model.model import Model
from dessn.toy.edges import ToCount, ToFlux, ToLuminosity, ToLuminosityDistance, ToRate, ToRedshift, ToType
from dessn.toy.latent import Luminosity, Redshift, Type
from dessn.toy.underlying import Cosmology, SupernovaIaDist, SupernovaIIDist, SupernovaRate
from dessn.toy.transformations import Flux, LuminosityDistance
from dessn.toy.observed import ObservedCounts, ObservedRedshift, ObservedType
from dessn.simulation.simulation import Simulation
import logging
import os


class ToyModel(Model):
    """ A modified toy model.


    .. figure::     ../plots/toyModelPGM.png
        :align:     center
    """
    def __init__(self, observations):
        super(ToyModel, self).__init__("ToyModel")

        z_o = observations["z_o"]
        z_o_err = observations["z_o_err"]
        count_o = observations["count_o"]
        type_o = observations["type_o"]
        n = z_o.size

        self.add_node(ObservedType(type_o))
        self.add_node(ObservedRedshift(z_o, z_o_err))
        self.add_node(ObservedCounts(count_o))

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
    dir_name = os.path.dirname(__file__)
    pgm_file = os.path.abspath(dir_name + "/../../plots/toyModelPGM.png")
    temp_dir = os.path.abspath(dir_name + "/../../temp/toyModel")
    plot_file = os.path.abspath(dir_name + "/../../plots/toyModelChain.png")
    walk_file = os.path.abspath(dir_name + "/../../plots/toyModelWalks.png")

    vals = [0.28, -1.0, 72, 10, 0.1, 5, 0.2, 0.6]
    simulation = Simulation()
    observations, theta = simulation.get_simulation(*vals, num_trans=50)
    toy_model = ToyModel(observations)
    # fig = toy_model.get_pgm(pgm_file)
    if True:
        toy_model.fit_model(num_steps=100, num_burn=50, temp_dir=temp_dir, save_interval=20)
        chain_consumer = toy_model.get_consumer()
        # chain_consumer.plot_walks(display=False, filename=walk_file, figsize=(8, 12))
        chain_consumer.configure_contour(cloud=False)
        chain_consumer.configure_general(bins=0.3)
        chain_consumer.plot(display=False, filename=plot_file, figsize="grow", truth=vals)

