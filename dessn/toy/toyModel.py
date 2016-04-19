from dessn.model.model import Model
from dessn.toy.edges import ToLightCurve, ToLuminosity, ToRate, ToRedshift, ToType
from dessn.toy.latent import Luminosity, Redshift, Type, Colour, PeakTime, Stretch
from dessn.toy.underlying import SupernovaRate, OmegaM, Hubble, SupernovaIaDist1, SupernovaIaDist2, SupernovaIIDist1, SupernovaIIDist2
from dessn.toy.observed import ObservedRedshift, ObservedType, ObservedLightCurves
from dessn.simulation.simulation import Simulation
import logging
import sys
import os


class ToyModel(Model):
    """ A modified toy model. The likelihood surfaces and PGM model are given below.

    Probabilities and model details are recorded in the model parameter and edge classes.


    .. figure::     ../plots/toyModelPGM.png
        :align:     center

    .. figure::     ../plots/toyModelChain.png
        :align:     center

    .. figure::     ../plots/toyModelWalks.png
        :align:     center

    """
    def __init__(self, observations):
        super(ToyModel, self).__init__("ToyModel")

        z_o = observations["z_o"]
        lcs_o = observations["lcs_o"]
        type_o = observations["type_o"]
        n = len(z_o)

        self.add_node(ObservedType(type_o))
        self.add_node(ObservedRedshift(z_o))
        self.add_node(ObservedLightCurves(lcs_o))

        self.add_node(OmegaM())
        self.add_node(Hubble())
        self.add_node(SupernovaIaDist1())
        self.add_node(SupernovaIaDist2())
        self.add_node(SupernovaIIDist1())
        self.add_node(SupernovaIIDist2())
        self.add_node(SupernovaRate())

        self.add_node(Luminosity(n=n))
        self.add_node(Stretch(n=n))
        self.add_node(PeakTime(n=n))
        self.add_node(Colour(n=n))
        self.add_node(Redshift(n=n))
        self.add_node(Type(n=n))

        self.add_edge(ToLightCurve())
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
    temp_dir = os.path.abspath(dir_name + "/../../temp/toyModel")
    plot_file = os.path.abspath(dir_name + "/../../plots/toyModelChain.png")
    walk_file = os.path.abspath(dir_name + "/../../plots/toyModelWalks.png")

    vals = {"num_days": 30, "omega_m": 0.28, "H0": 72, "snIa_luminosity": -19.3, "snIa_sigma": 0.1,
            "snII_luminosity": -18, "snII_sigma": 0.2, "sn_rate": 0.5}
    simulation = Simulation()
    observations, theta = simulation.get_simulation(**vals)
    toy_model = ToyModel(observations)
    '''
    if not only_data:
        np.random.seed(102)
        pgm_file = os.path.abspath(dir_name + "/../../plots/toyModelPGM.png")
        fig = toy_model.get_pgm(pgm_file)'''

    toy_model.fit_model(num_steps=5000, num_burn=500, temp_dir=temp_dir, save_interval=60)

    if not only_data:
        chain_consumer = toy_model.get_consumer()
        chain_consumer.configure_general(max_ticks=4, bins=0.7)
        chain_consumer.plot_walks(display=False, filename=walk_file, figsize=(20, 10), truth=theta)
        chain_consumer.plot(display=False, filename=plot_file, figsize="grow", truth=theta)

