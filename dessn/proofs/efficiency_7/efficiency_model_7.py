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


class ActualError(ParameterUnderlying):
    def __init__(self):
        super().__init__("sigma", r"$\sigma$")

    def get_log_prior(self, data):
        if data["sigma"] < 0:
            return -np.inf
        return 1

    def get_suggestion_requirements(self):
        return ["data"]

    def get_suggestion(self, data):
        return np.std(data["data"])

    def get_suggestion_sigma(self, data):
        return 10


class ToUnderlyingCorrected(Edge):
    def __init__(self, threshold):
        super().__init__(["data"], ["mean", "sigma"])
        self.factor = np.log(np.sqrt(2 * np.pi))
        self.threshold = threshold

    def get_log_likelihood(self, data):
        diff = data["data"] - data["mean"]
        sigma = data["sigma"]
        prob_happening = -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)

        bound = self.threshold - data["mean"]
        ltz = (bound < 0) * 2.0 - 1.0
        integral = ltz * 0.5 * erf((np.abs(bound)) / (np.sqrt(2) * sigma)) + 0.5
        log_integral = np.log(integral)
        result = prob_happening - log_integral
        result[result == np.inf] = -np.inf
        return result


class ToUnderlying(Edge):
    def __init__(self):
        super().__init__(["data"], ["mean", "sigma"])
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["data"] - data["mean"]
        sigma = data["sigma"]
        return -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)


class EfficiencyModelUncorrected(Model):
    def __init__(self, realised, seed=0, name="Uncorrected"):
        super().__init__(name)
        np.random.seed(seed)
        self.add(ObservedPoints(realised))
        self.add(ActualValue())
        self.add(ActualError())
        self.add(ToUnderlying())
        self.finalise()


class EfficiencyModelCorrected(Model):
    def __init__(self, realised, threshold, seed=0, name="Corrected"):
        super().__init__(name)
        np.random.seed(seed)
        self.add(ObservedPoints(realised))
        self.add(ActualValue())
        self.add(ActualError())
        self.add(ToUnderlyingCorrected(threshold))
        self.finalise()


def get_weights(alpha, mu, sigma, n):
    sign = np.sign(alpha - mu)
    abs = np.abs(alpha - mu)
    prob = 0.5 - sign * 0.5 * erf(abs / (np.sqrt(2) * sigma))
    prob = np.power(prob, n)
    return prob


def get_data(seed=5, n=2000):
    np.random.seed(seed=seed)
    mean = 100.0
    cut = 50
    sigma = 20
    observed = mean + np.random.normal(size=n) * sigma
    mask = observed > cut
    print(mask.sum(), n, observed.mean(), observed[mask].mean())
    return mean, sigma, cut, observed, mask


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(__file__)
    t = os.path.abspath(dir_name + "/output/data_%d")
    plot_file = os.path.abspath(dir_name + "/output/surfaces.png")
    walk_file = os.path.abspath(dir_name + "/output/walk_%s.png")

    c = ChainConsumer()
    n = 2
    colours = ["#4CAF50", "#D32F2F", "#1E88E5"] * n  # , "#FFA000"] * n
    for i in range(n):
        mean, sigma, cut, observed, mask = get_data(seed=i)

        model_good = EfficiencyModelUncorrected(observed, name="Good")
        model_un = EfficiencyModelUncorrected(observed[mask])
        model_cor = EfficiencyModelCorrected(observed[mask], cut)

        sampler = EnsembleSampler(num_steps=25000, num_burn=1000, temp_dir=t % i)
        model_good.fit(sampler, chain_consumer=c)
        model_un.fit(sampler, chain_consumer=c)
        biased_chain = c.chains[-1]
        # model_cor.fit(sampler, chain_consumer=c)

        mus = biased_chain[:, 0]
        sigmas = biased_chain[:, 1]
        weights = 1 / get_weights(cut, mus, sigmas, mask.sum())

        c.add_chain(biased_chain, name="Importance Sampled", weights=weights)

    c.configure_bar(shade=True)
    c.configure_general(colours=colours, bins=0.5)
    c.configure_contour(contourf=True, contourf_alpha=0.2)
    c.plot(filename=plot_file, figsize=(5, 5), truth=[mean, sigma], legend=False)
