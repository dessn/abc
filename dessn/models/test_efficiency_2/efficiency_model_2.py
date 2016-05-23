from dessn.framework.model import Model
from dessn.framework.edge import Edge
from dessn.framework.parameter import ParameterObserved, ParameterLatent, ParameterUnderlying
from dessn.framework.samplers.ensemble import EnsembleSampler

import numpy as np
import os
import logging
from scipy.special import erf


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
        return np.mean(data["data"]) * 0.5 + 0.5 * 100

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


class ToLatentUncorrected(Edge):
    def __init__(self):
        super().__init__(["data", "data_error"], "point")
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["data"] - data["point"]
        sigma = data["data_error"]
        return -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)


class ToLatentCorrected(Edge):
    def __init__(self, threshold):
        super().__init__(["data", "data_error"], ["point"])
        self.factor = np.log(np.sqrt(2 * np.pi))
        self.threshold = threshold

    def get_log_likelihood(self, data):
        diff = data["data"] - data["point"]
        sigma = data["data_error"]
        prob_happening = -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)

        bound = self.threshold * sigma - data["point"]
        ltz = (bound < 0) * 2.0 - 1.0
        integral = ltz * 0.5 * erf((np.abs(bound)) / (np.sqrt(2) * sigma)) + 0.5
        log_integral = np.log(integral)
        return prob_happening - log_integral


class ToUnderlying(Edge):
    def __init__(self):
        super().__init__("point", ["mean", "sigma"])
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["point"] - data["mean"]
        sigma = data["sigma"]
        return -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)


class EfficiencyModelUncorrected(Model):
    def __init__(self, realised, realised_errors, seed=0):
        super().__init__("Uncorrected")
        np.random.seed(seed)
        self.add_node(ObservedError(realised_errors))
        self.add_node(ObservedPoints(realised))
        self.add_node(ActualPoint(realised.size))
        self.add_node(ActualMean())
        self.add_node(ActualSigma())
        self.add_edge(ToLatentUncorrected())
        self.add_edge(ToUnderlying())
        self.finalise()


class EfficiencyModelCorrected(Model):
    def __init__(self, realised, realised_errors, threshold, seed=0):
        super().__init__("Corrected")
        np.random.seed(seed)
        self.add_node(ObservedError(realised_errors))
        self.add_node(ObservedPoints(realised))
        self.add_node(ActualPoint(realised.size))
        self.add_node(ActualMean())
        self.add_node(ActualSigma())
        self.add_edge(ToLatentCorrected(threshold))
        self.add_edge(ToUnderlying())
        self.finalise()


def get_data():
    np.random.seed(1)
    mean = 100.0
    std = 30.0
    alpha = 3.5
    n = 300

    actual = np.random.normal(loc=mean, scale=std, size=n)

    errors = np.ones(actual.shape) * mean / 5
    observed = actual + np.random.normal(size=n) * errors

    ston = observed / errors
    mask = ston > alpha
    print(mask.sum(), n, observed[mask].mean())

    return mean, std, observed[mask], errors[mask], alpha

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(__file__)
    t = os.path.abspath(dir_name + "/output/data_%s")
    plot_file = os.path.abspath(dir_name + "/output/surfaces.png")
    walk_file = os.path.abspath(dir_name + "/output/walk_%s.png")

    mean, std, observed, errors, alpha = get_data()

    model_un = EfficiencyModelUncorrected(observed, errors)
    model_cor = EfficiencyModelCorrected(observed, errors, alpha)

    pgm_file = os.path.abspath(dir_name + "/output/pgm.png")
    fig = model_cor.get_pgm(pgm_file)

    sampler = EnsembleSampler(num_steps=4250, num_burn=500, temp_dir=t % "no", save_interval=60)
    chain_un = model_un.fit(sampler)

    sampler.temp_dir = t % "cor"
    chain_consumer_cor = model_cor.fit(sampler)

    chain_consumer_cor.add_chain(chain_un.chains[0], name=chain_un.names[0])

    chain_consumer_cor.configure_bar(shade=True)
    chain_consumer_cor.configure_general(bins=0.7)
    # chain_consumer_cor.plot_walks(filename=walk_file % "no", chain=1, truth=[mean, std])
    chain_consumer_cor.plot_walks(filename=walk_file % "cor", chain=0, truth=[mean, std])
    # chain_consumer_cor.plot(filename=plot_file, figsize=(8, 8), truth=[mean, std])
