import logging
import os
import sys
import numpy as np

from dessn.models.a_pure_snia.edges import ToLightCurve, ToLuminosity, ToRedshift
from dessn.models.a_pure_snia.latent import Luminosity, Redshift, Colour, PeakTime, Stretch
from dessn.models.a_pure_snia.observed import ObservedRedshift, ObservedLightCurves
from dessn.models.a_pure_snia.underlying import OmegaM, Hubble, SupernovaIaDist1, SupernovaIaDist2

from dessn.framework.model import Model
from dessn.models.a_pure_snia.simulation import Simulation


class PureModel(Model):
    """ A modified mixed_types framework. The likelihood surfaces and PGM framework are given below.

    Probabilities and framework details are recorded in the framework parameter and edge classes.


    .. figure::     ../plots/toyModelPGM.png
        :align:     center

    .. figure::     ../plots/toyModelChain.png
        :align:     center

    .. figure::     ../plots/toyModelWalks.png
        :align:     center

    """
    def __init__(self, observations):
        super(PureModel, self).__init__("ToyModel")

        z_o = observations["z_o"]
        lcs_o = observations["lcs_o"]
        n = len(z_o)

        self.add_node(ObservedRedshift(z_o))
        self.add_node(ObservedLightCurves(lcs_o))

        self.add_node(OmegaM())
        self.add_node(Hubble())
        self.add_node(SupernovaIaDist1())
        self.add_node(SupernovaIaDist2())

        self.add_node(Luminosity(n=n))
        self.add_node(Stretch(n=n))
        self.add_node(PeakTime(n=n))
        self.add_node(Colour(n=n))
        self.add_node(Redshift(n=n))

        self.add_edge(ToLightCurve())
        self.add_edge(ToLuminosity())
        self.add_edge(ToRedshift())

        self.finalise()

if __name__ == "__main__":
    only_data = len(sys.argv) > 1
    if only_data:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    else:
        logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(__file__)
    temp_dir = os.path.abspath(dir_name + "/output/pure_snia")
    plot_file = os.path.abspath(dir_name + "/output/pure_snia.png")
    walk_file = os.path.abspath(dir_name + "/output/pure_snia_walk.png")

    vals = {"num_transient": 20, "omega_m": 0.5, "H0": 85, "snIa_luminosity": -19.3,
            "snIa_sigma": 0.001}
    simulation = Simulation()
    observations, theta = simulation.get_simulation(**vals)
    model = PureModel(observations)

    if not only_data:
        np.random.seed(102)
        pgm_file = os.path.abspath(dir_name + "/output/pure_snia.png")
        # fig = model.get_pgm(pgm_file)

    # print(model._theta_names, model._theta_labels)
    # t = model._get_suggestion()
    # t[model._theta_names.index("H0")] = 85
    # # print("OOO: ", observations["z_o"][0], observations["lcs_o"][0])
    # for om in np.linspace(0.1, 0.8, 15):
    #     t[model._theta_names.index("omega_m")] = om
    #     # print(om, model.get_log_posterior(t))
    #     print(om, model.get_log_likelihood(t))

    #
    # z = observations["z_o"][0]
    # import sncosmo
    # sncosmo_model = sncosmo.Model(source="salt2")
    # sncosmo_model.set(z=z)
    # print(observations["lcs_o"][0].meta)
    # res, fitted_model = sncosmo.fit_lc(observations["lcs_o"][0], sncosmo_model,
    #                                    ['t0', 'x0', 'x1', 'c'])
    # sncosmo.plot_lc(observations["lcs_o"][0], model=fitted_model, errors=res.errors, fname=dir_name+"/output/death.png")

    # edge = ToLightCurve()
    # print(theta)
    # oms = np.linspace(0.1, 0.7, 39)
    # vals = []
    # for om in oms:
    #     l = -np.inf
    #
    #     for lc, z in zip(observations["lcs_o"], observations["z_o"]):
    #         data = lc.meta.copy()
    #         data["redshift"] = z
    #         data["H0"] = 85
    #         data["luminosity"] = -19.3
    #         data["olc"] = lc
    #         data["omega_m"] = om
    #         l = np.logaddexp(l, edge.get_log_likelihood(data))
    #     vals.append(l)
    # vals = np.array(vals)
    # vals -= vals.max()
    # for om,v in zip(oms, vals):
    #     print(om, v)


    model.fit_model(num_steps=10000, num_burn=0, temp_dir=temp_dir, save_interval=60)

    if not only_data:
        chain_consumer = model.get_consumer()
        chain_consumer.configure_general(max_ticks=4, bins=0.7)
        chain_consumer.plot_walks(display=False, filename=walk_file, figsize=(20, 10), truth=theta)
        chain_consumer.plot(display=False, filename=plot_file, figsize="grow", truth=theta)

