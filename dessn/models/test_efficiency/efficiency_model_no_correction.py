from dessn.framework.model import Model
from dessn.framework.edge import Edge
from dessn.framework.parameter import ParameterObserved, ParameterLatent, ParameterUnderlying
from dessn.framework.samplers.ensemble import EnsembleSampler

import numpy as np
import os
import logging


class ObservedPoints(ParameterObserved):
    def __init__(self, data):
        super().__init__("data", "$d$", data)


class ObservedError(ParameterObserved):
    def __init__(self, data):
        super().__init__("data_error", "$e$", data)


class ActualPoint(ParameterLatent):
    def __init__(self, n):
        super().__init__("point", "$p$")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return ["data", "data_error"]

    def get_suggestion(self, data):
        return data["data"]

    def get_suggestion_sigma(self, data):
        return data["data_error"]


class ActualMean(ParameterUnderlying):
    def __init__(self):
        super().__init__("mean", r"$\mu$")

    def get_log_prior(self, data):
        return 1

    def get_suggestion_requirements(self):
        return ["data"]

    def get_suggestion(self, data):
        return np.mean(data["data"])

    def get_suggestion_sigma(self, data):
        return np.std(data["data"]) * 2


class ActualSigma(ParameterUnderlying):
    def __init__(self):
        super().__init__("sigma", r"$\sigma$")

    def get_suggestion(self, data):
        return np.std(data["data"])

    def get_suggestion_sigma(self, data):
        return 0.5 * np.std(data["data"])

    def get_suggestion_requirements(self):
        return ["data"]

    def get_log_prior(self, data):
        if data["sigma"] < 0:
            return -np.inf
        return 1


class ToLatent(Edge):
    def __init__(self):
        super().__init__(["data", "data_error"], "point")
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["data"] - data["point"]
        sigma = data["data_error"]
        return -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)


class ToUnderlying(Edge):
    def __init__(self):
        super().__init__("point", ["mean", "sigma"])
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["point"] - data["mean"]
        sigma = data["sigma"]
        return -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)


class EfficiencyModel(Model):
    def __init__(self, realised, realised_errors, seed=0):
        super().__init__("Efficiency")
        np.random.seed(seed)
        self.add_node(ObservedError(realised_errors))
        self.add_node(ObservedPoints(realised))
        self.add_node(ActualPoint(realised.size))
        self.add_node(ActualMean())
        self.add_node(ActualSigma())
        self.add_edge(ToLatent())
        self.add_edge(ToUnderlying())
        self.finalise()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(__file__)
    temp_dir = os.path.abspath(dir_name + "/output/data_no")
    plot_file = os.path.abspath(dir_name + "/output/surface_no.png")
    walk_file = os.path.abspath(dir_name + "/output/walk_no.png")

    np.random.seed(0)
    mean = 100.0
    std = 30.0
    threshold = 8
    n = 100

    actual = np.random.normal(loc=mean, scale=std, size=n)
    actual_errors = np.sqrt(actual)

    realised = actual + np.random.normal(loc=0, scale=actual_errors, size=n)
    realised_errors = np.sqrt(realised)

    ston = realised / realised_errors
    mask = ston > threshold

    realised = realised[mask]
    realised_errors = realised_errors[mask]
    print(mask.sum(), n, realised.mean())
    model = EfficiencyModel(realised, realised_errors)

    pgm_file = os.path.abspath(dir_name + "/output/pgm_no.png")
    # fig = model.get_pgm(pgm_file)

    sampler = EnsembleSampler(num_steps=4000, num_burn=500, temp_dir=temp_dir, save_interval=60)
    chain_consumer = model.fit(sampler)

    chain_consumer.configure_general(max_ticks=4, bins=0.5)
    # chain_consumer.plot_walks(display=False, filename=walk_file, truth=[mean, std])
    chain_consumer.plot(display=False, filename=plot_file, figsize="page", truth=[mean, std])