from dessn.framework.model import Model
from dessn.framework.edge import Edge
from dessn.framework.parameter import ParameterObserved, ParameterUnderlying
from dessn.framework.samplers.ensemble import EnsembleSampler
from chainconsumer import ChainConsumer
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


class ActualValue(ParameterUnderlying):
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


class ToUnderlyingCorrected(Edge):
    def __init__(self, threshold):
        super().__init__(["data", "data_error"], ["mean"])
        self.factor = np.log(np.sqrt(2 * np.pi))
        self.threshold = threshold

    def get_log_likelihood(self, data):
        diff = data["data"] - data["mean"]
        sigma = data["data_error"]
        prob_happening = -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)

        bound = self.threshold * sigma - data["mean"]
        ltz = (bound < 0) * 2.0 - 1.0
        integral = ltz * 0.5 * erf((np.abs(bound)) / (np.sqrt(2) * sigma)) + 0.5
        log_integral = np.log(integral)

        return prob_happening - log_integral


class ToUnderlying(Edge):
    def __init__(self):
        super().__init__(["data", "data_error"], ["mean"])
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["data"] - data["mean"]
        sigma = data["data_error"]
        return -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)


class EfficiencyModelUncorrected(Model):
    def __init__(self, realised, realised_errors, seed=0):
        super().__init__("Uncorrected")
        np.random.seed(seed)
        self.add_node(ObservedError(realised_errors))
        self.add_node(ObservedPoints(realised))
        self.add_node(ActualValue())
        self.add_edge(ToUnderlying())
        self.finalise()


class EfficiencyModelCorrected(Model):
    def __init__(self, realised, realised_errors, threshold, seed=0):
        super().__init__("Corrected")
        np.random.seed(seed)
        self.add_node(ObservedError(realised_errors))
        self.add_node(ObservedPoints(realised))
        self.add_node(ActualValue())
        self.add_edge(ToUnderlyingCorrected(threshold))
        self.finalise()


def get_data(seed=5):
    np.random.seed(seed=seed)
    mean = 100.0
    alpha = 4
    n = 1000
    errors = np.ones(n) * mean / 5
    observed = mean + np.random.normal(size=n) * errors
    ston = observed / errors
    mask = ston > alpha
    observed = observed[mask]
    errors = errors[mask]
    print(mask.sum(), n, observed.mean())
    return mean, observed, errors, alpha


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(__file__)
    t = os.path.abspath(dir_name + "/output/run_%d")
    plot_file = os.path.abspath(dir_name + "/output/surfaces.png")
    walk_file = os.path.abspath(dir_name + "/output/walk_%s.png")

    c = ChainConsumer()
    n = 3
    colours = ["#D32F2F", "#1E88E5"] * n
    for i in range(n):
        mean, observed, errors, alpha = get_data(seed=i)

        model_un = EfficiencyModelUncorrected(observed, errors)
        model_cor = EfficiencyModelCorrected(observed, errors, alpha)

        pgm_file = os.path.abspath(dir_name + "/output/pgm.png")
        fig = model_cor.get_pgm(pgm_file)

        sampler = EnsembleSampler(num_steps=5000, num_burn=1000, temp_dir=t % i, num_walkers=50)
        model_un.fit(sampler, chain_consumer=c)
        model_cor.fit(sampler, chain_consumer=c)

    c.configure_bar(shade=True)
    c.configure_general(colours=colours)
    c.plot(filename=plot_file, figsize=(5, 3), truth=[mean], legend=False)
