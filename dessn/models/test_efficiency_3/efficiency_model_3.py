from dessn.framework.model import Model
from dessn.framework.edge import Edge
from dessn.framework.parameter import ParameterObserved, ParameterLatent, ParameterUnderlying
from dessn.framework.samplers.batch import BatchMetroploisHastings
from dessn.framework.samplers.metropolisHastings import MetropolisHastings
from dessn.framework.samplers.ensemble import EnsembleSampler
from dessn.chain.chain import ChainConsumer
from dessn.utility.viewer import Viewer

import numpy as np
import os
import logging
from scipy.special import erf
from scipy.integrate import simps


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


class ToLatent(Edge):
    def __init__(self):
        super().__init__(["data", "data_error"], "point")
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["data"] - data["point"]
        sigma = data["data_error"]
        return -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)


class BiasCorrection(Edge):
    def __init__(self, threshold):
        super().__init__(["data_error", "point"], ["mean", "sigma"])
        self.factor = np.log(np.sqrt(2 * np.pi))
        self.threshold = threshold
        self.arr = np.linspace(-4, 4, 300)
        self.gauss = (1 / (np.sqrt(2 * np.pi))) * np.exp(-(self.arr * self.arr) / 2)

    def get_log_likelihood(self, data):
        mu = data["mean"]
        sigma = data["sigma"]
        shape = data["point"].shape
        error = data["data_error"][0]

        s = self.arr * sigma + mu
        bound = self.threshold * error - s
        ltz = (bound < 0) * 2.0 - 1.0
        integral = ltz * 0.5 * erf((np.abs(bound)) / (np.sqrt(2) * error)) + 0.5

        multiplied = integral * self.gauss
        denom = simps(multiplied) / (self.arr[-1] - self.arr[0])
        result = -np.log(denom)
        if result == np.inf: result = -np.inf
        return np.ones(shape) * result


class ToUnderlying(Edge):
    def __init__(self):
        super().__init__("point", ["mean", "sigma"])
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["point"] - data["mean"]
        sigma = data["sigma"]
        return -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)


class EfficiencyModelUncorrected(Model):
    def __init__(self, realised, realised_errors, seed=0, name="Uncorrected"):
        super().__init__(name)
        np.random.seed(seed)
        self.add_node(ObservedError(realised_errors))
        self.add_node(ObservedPoints(realised))
        self.add_node(ActualPoint(realised.size))
        self.add_node(ActualMean())
        self.add_node(ActualSigma())
        self.add_edge(ToLatent())
        self.add_edge(ToUnderlying())
        self.finalise()


class EfficiencyModelCorrected(Model):
    def __init__(self, realised, realised_errors, threshold, seed=0, name="Corrected"):
        super().__init__(name)
        np.random.seed(seed)
        self.add_node(ObservedError(realised_errors))
        self.add_node(ObservedPoints(realised))
        self.add_node(ActualPoint(realised.size))
        self.add_node(ActualMean())
        self.add_node(ActualSigma())
        self.add_edge(ToLatent())
        self.add_edge(BiasCorrection(threshold))
        self.add_edge(ToUnderlying())
        self.finalise()


def get_data(seed=5):
    np.random.seed(seed=seed)
    mean = 100.0
    std = 20.0
    alpha = 3.75
    n = 200

    actual = np.random.normal(loc=mean, scale=std, size=n)

    errors = np.ones(actual.shape) * 20
    observed = actual + np.random.normal(size=n) * errors

    ston = observed / errors
    mask = ston > alpha
    print(mask.sum(), n, observed[mask].mean())

    return mean, std, observed[mask], errors[mask], alpha, actual, observed, errors

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dir_name = os.path.dirname(__file__)
    t = os.path.abspath(dir_name + "/output/data_%s")
    plot_file = os.path.abspath(dir_name + "/output/surfaces.png")
    walk_file = os.path.abspath(dir_name + "/output/walk_%s.png")

    # model_un = EfficiencyModelUncorrected(np.random.random(10), np.random.random(10))
    # pgm_file = os.path.abspath(dir_name + "/output/pgm.png")
    # fig = model_un.get_pgm(pgm_file)

    c = ChainConsumer()
    v = Viewer([[80, 120], [10, 40]], parameters=[r"$\mu$", r"$\sigma$"], truth=[100, 20])

    n = 1
    colours = ["#4CAF50", "#2C6F30", "#D32F2F","#732F2F", "#1E88E5", "#1E5895"] * n
    for i in range(n):
        mean, std, observed, errors, alpha, actual, uo, oe = get_data(seed=i)
        theta = [mean, std] + actual.tolist()

        model_un = EfficiencyModelUncorrected(observed, errors, name="Uncorrected")
        model_good = EfficiencyModelUncorrected(uo, oe, name="Good")
        model_cor = EfficiencyModelCorrected(observed, errors, alpha, name="Corrected")

        kwargs = {"num_steps": 200000, "num_burn": 40000}
        sampler = BatchMetroploisHastings(num_walkers=8, kwargs=kwargs, temp_dir=t % i, num_cores=1)
        sampler2 = EnsembleSampler(temp_dir=t % i)

        model_good.fit(sampler, chain_consumer=c)
        model_good.fit(sampler2, chain_consumer=c)
        model_un.fit(sampler, chain_consumer=c)
        model_un.fit(sampler2, chain_consumer=c)
        model_cor.fit(sampler, chain_consumer=c)
        model_cor.fit(sampler2, chain_consumer=c)
        print("Good ", model_un.get_log_posterior(theta), c.chains[-3][-1, 0])
        print("Uncorrected ", model_un.get_log_posterior(theta), c.chains[-2][-1, 0])
        print("Corrected ", model_cor.get_log_posterior(theta), c.chains[-1][-1, 0])

    c.configure_bar(shade=True)
    c.configure_general(bins=1.0, colours=colours)
    c.configure_contour(sigmas=[0, 0.01, 1, 2], contourf=True, contourf_alpha=0.3)
    # c.plot_walks(filename=walk_file % "good", chain=0, truth=[mean, std])
    # c.plot_walks(filename=walk_file % "un", chain=1, truth=[mean, std])
    # c.plot_walks(filename=walk_file % "cor", chain=1, truth=[mean, std])
    c.plot(filename=plot_file, figsize=(8, 8), truth=[mean, std], legend=False)
