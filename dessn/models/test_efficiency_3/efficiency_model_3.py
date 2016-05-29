from dessn.framework.model import Model
from dessn.framework.edge import Edge
from dessn.framework.parameter import ParameterObserved, ParameterLatent, ParameterUnderlying
from dessn.framework.samplers.batch import BatchMetroploisHastings
from dessn.framework.samplers.metropolisHastings import MetropolisHastings
from dessn.framework.samplers.ensemble import EnsembleSampler
from dessn.framework.samplers.nestled import NestledSampler
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
        super().__init__("point", "$s$")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return ["data", "data_error"]

    def get_suggestion(self, data):
        return data["data"]

    def get_suggestion_sigma(self, data):
        return 3 * data["data_error"]


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
        return np.std(data["data"])

    def get_suggestion_requirements(self):
        return ["data"]

    def get_log_prior(self, data):
        if data["sigma"] < 0:
            return -np.inf
        g = 20
        s = data["sigma"]
        return -2 * np.log(s)
        # return np.log((2 / (np.pi * g)) * (g * g / (s * s + g * g)))
        # return 1


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
        self.arr = np.linspace(-4, 4, 200)
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


def get_data(seed=5, n=3000):
    np.random.seed(seed=seed)
    mean = 100.0
    std = 20.0
    alpha = 3.25

    actual = np.random.normal(loc=mean, scale=std, size=n)

    errors = np.ones(actual.shape) * 20
    observed = actual + np.random.normal(size=n) * errors

    ston = observed / errors
    mask = ston > alpha
    print(mask.sum(), n, observed[mask].mean())

    return mean, std, observed[mask], errors[mask], alpha, actual, observed, errors, actual[mask]

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
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
    w = 4
    colours = ["#4CAF50", "#D32F2F", "#1E88E5"] * n * 2

    import matplotlib.pyplot as plt
    #
    # mean, std, observed, errors, alpha, actual, uo, oe = get_data(seed=100*(1+1))
    # plt.hist(observed, histtype="step")
    #
    # mean, std, observed, errors, alpha, actual, uo, oe = get_data(seed=100*(0+1))
    # plt.hist(observed, histtype="step")

    # plt.show()

    # mean, std, observed, errors, alpha, actual, uo, oe = get_data(seed=100 * (1 + 1))
    #
    # x = np.linspace(90, 110, 20)
    # y = np.linspace(10, 30, 20)
    # E = BiasCorrection(2)
    # xx, yy = np.meshgrid(x, y, sparse=True)
    # z = []
    # for x1 in x:
    #     for y1 in y:
    #         z.append(np.sum(E.get_log_likelihood({"mean":x1, "sigma":y1, "data": observed, "point": actual, "data_error":errors})))
    # z = np.array(z)
    # print(z.size)
    # z = z.reshape((20,20))
    # h = plt.contourf(x, y, z)
    # cbar = plt.colorbar(h)
    #
    # plt.show()
    # exit()
    for i in range(n):
        mean, std, observed, errors, alpha, actual, uo, oe, am = get_data(seed=i)
        theta_good = [mean, std] + actual.tolist()
        theta_bias = [mean, std] + am.tolist()
        kwargs = {"num_steps": 6000, "num_burn": 6000, "save_interval": 300}
        sampler = BatchMetroploisHastings(num_walkers=w, kwargs=kwargs, temp_dir=t % i, num_cores=4)
        # sampler = MetropolisHastings(**kwargs, temp_dir=t % i)
        # sampler2 = EnsembleSampler(temp_dir=t % i)
        # sampler3 = NestledSampler(temp_dir=t % i)

        model_good = EfficiencyModelUncorrected(uo, oe, name="Good%d" % i)
        model_good.fit(sampler, chain_consumer=c)
        print("Good ", model_good.get_log_posterior(theta_good), c.posteriors[-1][-1])

        model_un = EfficiencyModelUncorrected(observed, errors, name="Uncorrected%d" % i)
        model_un.fit(sampler, chain_consumer=c)
        print("Uncorrected ", model_un.get_log_posterior(theta_bias), c.posteriors[-1][-1])
        #
        model_cor = EfficiencyModelCorrected(observed, errors, alpha, name="Corrected%d" % i)
        model_cor.fit(sampler, chain_consumer=c)
        print("Corrected ", model_cor.get_log_posterior(theta_bias), c.posteriors[-1][-1])

        # model_good.fit(sampler3, chain_consumer=c, start=theta_good)
        # model_un.fit(sampler3, chain_consumer=c, start=theta_bias)
        # model_cor.fit(sampler3, chain_consumer=c, start=theta_bias)

    c.configure_bar(shade=True)
    c.configure_general(bins=1.0, colours=colours)
    c.configure_contour(sigmas=[0, 0.01, 1, 2], contourf=True, contourf_alpha=0.3)
    c.plot(filename=plot_file, truth=theta_bias, figsize=(8, 8), legend=False)
    for i in range(len(c.chains)):
        c.plot_walks(filename=walk_file % c.names[i], chain=i, truth=[mean, std])
        c.divide_chain(i, w).configure_general(rainbow=True) \
            .plot(figsize=(8, 8), filename=plot_file.replace(".png", "_%s.png" % c.names[i]), truth=theta_bias)
