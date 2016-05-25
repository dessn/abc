from dessn.framework.model import Model
from dessn.framework.edge import Edge
from dessn.framework.parameter import ParameterObserved, ParameterLatent, ParameterUnderlying
from dessn.framework.samplers.ensemble import EnsembleSampler
from dessn.chain.chain import ChainConsumer

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
        return 105

    def get_suggestion_sigma(self, data):
        return np.std(data["data"])


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
        result = prob_happening - log_integral
        result[result == np.inf] = -np.inf
        return result


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


def get_data(seed=5):
    np.random.seed(seed=seed)
    mean = 100.0
    std = 20.0
    alpha = 3.5
    n = 100

    actual = np.random.normal(loc=mean, scale=std, size=n)

    errors = np.ones(actual.shape) * 20
    observed = actual + np.random.normal(size=n) * errors

    ston = observed / errors
    mask = ston > alpha
    print(mask.sum(), n, observed[mask].mean())

    return mean, std, observed[mask], errors[mask], alpha

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(__file__)
    t = os.path.abspath(dir_name + "/output")
    t2 = os.path.abspath(dir_name + "/output/data_%s")
    plot_file = os.path.abspath(dir_name + "/output/surfaces.png")
    walk_file = os.path.abspath(dir_name + "/output/walk_%s.png")

    c = ChainConsumer()

    mean, std, observed, errors, alpha = get_data()

    from dessn.samplers.MetropolisHastings import MetropolisHastings

    model_un = EfficiencyModelUncorrected(observed, errors)

    sampler = EnsembleSampler(num_steps=9000, num_burn=400, temp_dir=t2 % "no")
    chain = model_un.fit(sampler)
    c.add_chain(chain.chains[0], name=chain.names[0], parameters=chain.default_parameters)

    sampler = MetropolisHastings(model_un.get_log_posterior, model_un.get_starting_position,
                                 uid="no", temp_dir=t, save_dims=model_un._num_actual,
                                 num_steps=90000, num_burn=30000)
    chain, weights = sampler.fit()
    c.add_chain(chain, weights=weights, name="Uncorrected MH")
    #
    model_cor = EfficiencyModelCorrected(observed, errors, alpha)

    sampler = EnsembleSampler(num_steps=9000, num_burn=400, temp_dir=t2 % "cor")
    chain = model_cor.fit(sampler)
    c.add_chain(chain.chains[0], name=chain.names[0], parameters=chain.default_parameters)


    sampler = MetropolisHastings(model_cor.get_log_posterior, model_cor.get_starting_position,uid="cor",
                                 temp_dir=t, save_dims=model_cor._num_actual,
                                 num_steps=90000, num_burn=30000)

    chain, weights = sampler.fit()
    c.add_chain(chain, weights=weights, name="Corrected MH")

    c.configure_bar(shade=True)
    # c.configure_general(bins=0.7, colours=colours)
    c.configure_contour(contourf=True, contourf_alpha=0.3)

    # c.plot_walks(filename=walk_file % "no", chain=1, truth=[mean, std])
    # c.plot_walks(filename=walk_file % "cor", chain=0, truth=[mean, std])
    c.plot(filename=plot_file, figsize=(8, 8), truth=[mean, std])
