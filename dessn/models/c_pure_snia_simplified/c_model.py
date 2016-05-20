import logging
import os
import sys
import numpy as np

from dessn.models.c_pure_snia_simplified.edges import ToRedshift, ToDistanceModulus, \
    ToParameters, ToMus, ToObservedDistanceModulus
from dessn.models.c_pure_snia_simplified.latent import ApparentMagnitude, Redshift, Colour, \
    Stretch, CosmologicalDistanceModulus, ObservedDistanceModulus
from dessn.models.c_pure_snia_simplified.observed import ObservedRedshift, ObservedC, \
    ObservedCovariance, ObservedMB, ObservedX1, ObservedInvCovariance
from dessn.models.c_pure_snia_simplified.underlying import OmegaM, Hubble, \
    IntrinsicScatter, Magnitude, AlphaStretch, BetaColour
from dessn.framework.samplers.ensemble import EnsembleSampler
from dessn.framework.samplers.polychord import PolyChord
from dessn.framework.model import Model
from dessn.models.c_pure_snia_simplified.simulation import Simulation


class PureModelSimple(Model):
    """ A modified mixed_types framework. The likelihood surfaces and PGM framework are given below.

    Probabilities and framework details are recorded in the framework parameter and edge classes.

    """
    def __init__(self, observations):
        super(PureModelSimple, self).__init__("ToyModel")

        z_o = observations["z_o"]
        n = len(z_o)

        self.add_node(ObservedRedshift(z_o))
        self.add_node(ObservedX1(observations["x1s"]))
        self.add_node(ObservedC(observations["cs"]))
        self.add_node(ObservedInvCovariance(observations["icovs"]))
        self.add_node(ObservedCovariance(observations["covs"]))
        self.add_node(ObservedMB(observations["mbs"]))

        self.add_node(OmegaM())
        self.add_node(Hubble())
        self.add_node(IntrinsicScatter())
        self.add_node(Magnitude())
        self.add_node(AlphaStretch())
        self.add_node(BetaColour())

        self.add_node(ApparentMagnitude(n=n))
        self.add_node(Stretch(n=n))
        self.add_node(Colour(n=n))
        self.add_node(Redshift(n=n))
        self.add_node(CosmologicalDistanceModulus())
        self.add_node(ObservedDistanceModulus())

        self.add_edge(ToDistanceModulus())
        self.add_edge(ToRedshift())
        self.add_edge(ToParameters())
        self.add_edge(ToMus())
        self.add_edge(ToObservedDistanceModulus())
        self.finalise()

if __name__ == "__main__":
    only_data = len(sys.argv) > 1
    if only_data:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    else:
        logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(__file__)
    temp_dir = os.path.abspath(dir_name + "/output/data")
    plot_file = os.path.abspath(dir_name + "/output/surface.png")
    walk_file = os.path.abspath(dir_name + "/output/walk.png")

    vals = {"num_transient": 100, "omega_m": 0.31, "H0": 65, "snIa_luminosity": -19.3,
            "snIa_sigma": 0.01, "alpha": 0.3, "beta": 2.0}
    simulation = Simulation()
    observations, theta = simulation.get_simulation(**vals)
    model = PureModelSimple(observations)

    if not only_data:
        np.random.seed(102)
        pgm_file = os.path.abspath(dir_name + "/output/pgm.png")
        # fig = model.get_pgm(pgm_file)

    sampler = EnsembleSampler(num_steps=6000, num_burn=1000, temp_dir=temp_dir, save_interval=60)
    model.fit(sampler)

    if not only_data:
        chain_consumer = model.get_consumer()
        chain_consumer.configure_general(max_ticks=4, bins=0.4)
        chain_consumer.plot_walks(display=False, filename=walk_file, figsize=(20, 10), truth=theta)
        chain_consumer.plot(display=False, filename=plot_file, figsize="grow", truth=theta)

