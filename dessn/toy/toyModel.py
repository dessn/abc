from dessn.model.model import Model
from dessn.toy.edges import ToCount, ToFlux, ToLuminosity, ToLuminosityDistance, ToRate, ToRedshift, ToType
from dessn.toy.latent import Luminosity, Redshift, Type
from dessn.toy.underlying import SupernovaRate, OmegaM, W, Hubble, SupernovaIaDist1, SupernovaIaDist2, SupernovaIIDist1, SupernovaIIDist2
from dessn.toy.transformations import Flux, LuminosityDistance
from dessn.toy.observed import ObservedCounts, ObservedRedshift, ObservedType
from dessn.simulation.simulation import Simulation
import logging
import sys
import os
import numpy as np

class ToyModel(Model):
    """ A modified toy model.


    .. figure::     ../plots/toyModelPGM.png
        :align:     center
    """
    def __init__(self, observations):
        super(ToyModel, self).__init__("ToyModel")

        z_o = observations["z_o"]
        count_o = observations["count_o"]
        type_o = observations["type_o"]
        n = len(z_o)

        self.add_node(ObservedType(type_o))
        self.add_node(ObservedRedshift(z_o))
        self.add_node(ObservedCounts(count_o))

        self.add_node(Flux())
        self.add_node(LuminosityDistance())

        self.add_node(OmegaM())
        # self.add_node(W())
        self.add_node(Hubble())
        self.add_node(SupernovaIaDist1())
        self.add_node(SupernovaIaDist2())
        self.add_node(SupernovaIIDist1())
        self.add_node(SupernovaIIDist2())
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
    only_data = len(sys.argv) > 1
    if only_data:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    else:
        logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(__file__)
    temp_dir = os.path.abspath(dir_name + "/../../temp/toyModel4")
    plot_file = os.path.abspath(dir_name + "/../../plots/toyModelChain4.png")
    walk_file = os.path.abspath(dir_name + "/../../plots/toyModelWalks4.png")

    vals = [0.28, -1.0, 72, 10, 0.01, 9.5, 0.02, 0.5]
    simulation = Simulation()
    observations, theta = simulation.get_simulation(*vals, num_trans=20)
    toy_model = ToyModel(observations)

    if not only_data:
        np.random.seed(0)
        pgm_file = os.path.abspath(dir_name + "/../../plots/toyModelPGM4.png")
        #fig = toy_model.get_pgm(pgm_file)

    toy_model.fit_model(num_steps=20000, num_burn=4000, temp_dir=temp_dir, save_interval=60)

    if not only_data:
        chain_consumer = toy_model.get_consumer()
        chain_consumer.plot_walks(display=False, filename=walk_file, figsize=(10, 6))
        chain_consumer.plot(display=False, filename=plot_file, figsize="grow", truth=[0.28, 72, 10, 0.01, 9.5, 0.02, 0.5])

