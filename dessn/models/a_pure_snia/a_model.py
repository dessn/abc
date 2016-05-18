import logging
import os
import sys
import numpy as np

from dessn.models.a_pure_snia.edges import ToLightCurve, ToRedshift, \
    ToAbsoluteMagnitude, ToDeltaM, ToDistanceModulus
from dessn.models.a_pure_snia.latent import Redshift, Colour, PeakTime, Stretch, \
    DeltaMag, AbsMag, DistanceModulus
from dessn.models.a_pure_snia.observed import ObservedRedshift, ObservedLightCurves
from dessn.models.a_pure_snia.underlying import OmegaM, Hubble, AbsoluteMagnitude, Scatter, \
    AlphaStretch, BetaColour
from dessn.framework.samplers.ensemble import EnsembleSampler
from dessn.framework.samplers.polychord import PolyChord

from dessn.framework.model import Model
from dessn.models.a_pure_snia.simulation import Simulation


class PureModel(Model):
    def __init__(self, o):
        super(PureModel, self).__init__("ToyModel")

        z_o = o["z"]
        n = len(z_o)

        self.add_node(ObservedRedshift(z_o))
        self.add_node(ObservedLightCurves(o["lcs"]))

        self.add_node(OmegaM())
        self.add_node(Hubble())
        self.add_node(AbsoluteMagnitude())
        self.add_node(Scatter())
        self.add_node(AlphaStretch())
        self.add_node(BetaColour())
        self.add_node(Stretch(n, o["x1"], o["x1s"]))
        self.add_node(PeakTime(n, o["t0"], o["t0s"]))
        self.add_node(Colour(n, o["c"], o["cs"]))
        self.add_node(Redshift(n))
        self.add_node(DeltaMag(n))
        self.add_node(AbsMag())
        self.add_node(DistanceModulus())

        self.add_edge(ToLightCurve())
        self.add_edge(ToRedshift())
        self.add_edge(ToAbsoluteMagnitude())
        self.add_edge(ToDeltaM())
        self.add_edge(ToDistanceModulus())

        self.finalise()

if __name__ == "__main__":
    only_data = len(sys.argv) > 1
    if only_data:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    else:
        logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(__file__)
    temp_dir = os.path.abspath(dir_name + "/output/data")
    pdir = os.path.abspath(dir_name + "/output")
    plot_file = os.path.abspath(dir_name + "/output/surface.png")
    walk_file = os.path.abspath(dir_name + "/output/walk.png")

    vals = {"num_transient": 15, "omega_m": 0.5, "H0": 85, "snIa_luminosity": -19.3,
            "snIa_sigma": 0.01}
    simulation = Simulation()
    observations, theta = simulation.get_simulation(**vals)
    model = PureModel(observations)

    if not only_data:
        np.random.seed(1)
        pgm_file = os.path.abspath(dir_name + "/output/pgm.png")
        # fig = model.get_pgm(pgm_file)

    sampler = EnsembleSampler(num_steps=2500, num_burn=200, temp_dir=temp_dir, save_interval=60)
    poly_sampler = PolyChord(pdir)
    model.fit(poly_sampler)

    if not only_data:
        chain_consumer = model.get_consumer()
        chain_consumer.configure_general(max_ticks=4, bins=0.7)
        chain_consumer.plot_walks(display=False, filename=walk_file, figsize=(20, 10), truth=theta)
        chain_consumer.plot(display=False, filename=plot_file, figsize="grow", truth=theta)

