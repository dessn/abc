import os
import logging

from dessn.framework.fitter import Fitter
from dessn.framework.models.approx_model import ApproximateModel
from dessn.framework.simulations.simple import SimpleSimulation

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(os.path.abspath(__file__)) + "/output/" + os.path.basename(__file__)[:-3]
    file = os.path.abspath(__file__)
    print(dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    model = ApproximateModel(100)
    # Turn off mass and skewness for easy test
    simulation = SimpleSimulation(alpha_c=0, dscale=0)

    fitter = Fitter(dir_name)
    fitter.set_models(model)
    fitter.set_simulations(simulation)

    fitter.fit(file)

