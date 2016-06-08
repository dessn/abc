from dessn.framework.model import Model
from dessn.framework.edge import Edge, EdgeTransformation
from dessn.framework.parameter import ParameterObserved, ParameterLatent, ParameterUnderlying, \
    ParameterTransformation
from dessn.framework.samplers.batch import BatchMetroploisHastings
from dessn.chain.chain import ChainConsumer
from dessn.utility.viewer import Viewer
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from scipy.special import erf
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator


class ObservedFlux(ParameterObserved):
    def __init__(self, data):
        super().__init__("f_o", r"$\vec{f_o}$", data)


class ObservedRedshift(ParameterObserved):
    def __init__(self, data):
        super().__init__("z_o", "$z_o$", data)


class LatentFlux(ParameterLatent):
    def __init__(self, n):
        super().__init__("f", "$f$")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion_requirements(self):
        return ["f_o"]

    def get_suggestion(self, data):
        return data["f_o"].mean(axis=1)

    def get_suggestion_sigma(self, data):
        return 0.5 * data["f_o"].mean(axis=1)


class Luminosity(ParameterTransformation):
    def __init__(self):
        super().__init__("L", "$L$")


class ActualMean(ParameterUnderlying):
    def __init__(self):
        super().__init__("mu", r"$\mu$")

    def get_log_prior(self, data):
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 200

    def get_suggestion_sigma(self, data):
        return 50


class ActualSigma(ParameterUnderlying):
    def __init__(self):
        super().__init__("sigma", r"$\sigma$")

    def get_suggestion_requirements(self):
        return ["f_o", "z_o"]

    def get_suggestion(self, data):
        return 40

    def get_suggestion_sigma(self, data):
        return 10

    def get_log_prior(self, data):
        if data["sigma"] < 0:
            return -np.inf
        return 1


class ToLatentFlux(Edge):
    def __init__(self):
        super().__init__(["f_o"], "f")
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["f_o"] - data["f"][:, None]
        sigma = np.sqrt(data["f"])
        sigma = sigma * sigma
        sigma = sigma[:, None]
        res = -0.5 * diff * diff / sigma - self.factor - np.log(sigma)
        res[np.isnan(res)] = -np.inf
        res = res.sum(axis=1)
        return res


class ToLuminosityCorrection(Edge):
    def __init__(self):
        super().__init__("L", "z_o")

    def get_log_likelihood(self, data):
        return 2 * np.log(1 + data["z_o"])


class ToLuminosity(EdgeTransformation):
    def __init__(self):
        super().__init__("L", ["f", "z_o"])

    def get_transformation(self, data):
        return {"L": data["f"] * (1 + data["z_o"]) * (1 + data["z_o"])}


class ToUnderlying(Edge):
    def __init__(self):
        super().__init__("L", ["mu", "sigma"])
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["L"] - data["mu"]
        sigma = data["sigma"]
        return -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)


class BiasCorrection(Edge):
    def __init__(self, threshold, number, temp_dir):
        super().__init__(["z_o"], ["mu", "sigma"])
        self.filename = temp_dir + os.sep + "bias_correction_%d.npy" % threshold
        self.threshold = threshold
        self.mus = None
        self.sigmas = None
        self.zs = None
        self.vs = None
        self.ms = None
        self.ss = None
        self.z = None
        self.N = number
        self.interp = self.get_data()

    def get_data(self):
        self.mus = np.linspace(100, 300, 50)
        self.sigmas = np.linspace(5, 100, 50)
        ms, ss = np.meshgrid(self.mus, self.sigmas, indexing="ij")
        self.ms = ms
        self.ss = ss
        if os.path.exists(self.filename):
            self.vs = np.load(self.filename)
        else:
            fs = np.linspace(1, 500, 500)
            bound = fs - self.threshold
            ltz = (bound > 0) * 2.0 - 1.0
            gplus = ltz * 0.5 * erf((np.abs(bound)) / (np.sqrt(2 * fs))) + 0.5
            gminus = 1 - gplus
            term = np.power(gminus, self.N - 1) * (gminus + self.N * gplus)
            z1 = 0.0
            z2 = 1.5
            zrange = 1 / (z2 - z1)
            zs = np.linspace(z1, z2, 100)
            self.vs = np.zeros(ms.shape)
            for i, m in enumerate(self.mus):
                for j, s in enumerate(self.sigmas):
                    # m = 100
                    # s = 40
                    zvals = np.zeros(zs.shape)
                    for k, z in enumerate(zs):
                        g = (fs * (1 + z) * (1 + z) - m)
                        gaussian = (1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-g * g / (2 * s * s))
                        integral = simps(gaussian * term, x=fs)
                        zvals[k] = integral
                    self.vs[i, j] = 1 - zrange *simps((1 + zs) * (1 + zs) * zvals, x=zs)
                    # print(1 - self.vs[i, j])
                    # plt.plot(zs, zvals)
                    # plt.show()
                    # exit()
            np.save(self.filename, self.vs)
        return RegularGridInterpolator((self.mus, self.sigmas), self.vs,
                                       bounds_error=False, fill_value=0.0)

    def get_log_likelihood(self, data):
        mu = data["mu"]
        sigma = data["sigma"]
        redshift = data["z_o"]

        points = [[mu, sigma]]
        p = self.interp(points)
        ps = -np.log(p)
        ps[ps == np.inf] = -np.inf
        return np.ones(redshift.shape) * ps


class EfficiencyModelUncorrected(Model):
    def __init__(self, observed_flux, observed_redshift, seed=0, name="Uncorrected"):
        super().__init__(name)
        np.random.seed(seed)
        self.add(ObservedRedshift(observed_redshift))
        self.add(ObservedFlux(observed_flux))
        self.add(LatentFlux(observed_redshift.size))
        self.add(ActualMean())
        self.add(ActualSigma())
        self.add(Luminosity())
        self.add(ToLuminosity())
        self.add(ToLatentFlux())
        self.add(ToUnderlying())
        self.add(ToLuminosityCorrection())
        self.finalise()


class EfficiencyModelCorrected(Model):
    def __init__(self, observed_flux, observed_redshift, threshold, number, temp_dir, seed=0, name="Corrected"):
        super().__init__(name)
        np.random.seed(seed)
        self.add(ObservedRedshift(observed_redshift))
        self.add(ObservedFlux(observed_flux))
        self.add(LatentFlux(observed_redshift.size))
        self.add(ActualMean())
        self.add(ActualSigma())
        self.add(Luminosity())
        self.add(ToLatentFlux())
        self.add(ToLuminosity())
        self.add(BiasCorrection(threshold, number, temp_dir))
        self.add(ToUnderlying())
        self.add(ToLuminosityCorrection())
        self.finalise()


def get_data(seed=5, n=1000):
    np.random.seed(seed=seed+1)
    num_obs = 7
    mean = 200.0
    std = 40.0
    z_start = 0
    z_end = 1.5
    threshold = 75

    z_o = np.random.uniform(z_start, z_end, size=n)
    lum = np.random.normal(loc=mean, scale=std, size=n)
    flux = lum / (1 + z_o)**2
    f_o = np.vstack((flux + np.random.normal(scale=np.sqrt(flux), size=n) for i in range(num_obs))).T
    obs_mask = f_o > threshold
    mask = obs_mask.sum(axis=1) >= 2
    print(mask.sum(), n, lum.mean(), lum[mask].mean())

    return mean, std, threshold, lum, z_o, f_o, mask, num_obs


def plot_weights(dir_name):
    mean, std, threshold, lall, zall, fall, mask, num_obs = get_data(seed=0)

    bias = BiasCorrection(threshold, num_obs, dir_name + "/output")
    m = bias.ms
    s = bias.ss
    v = bias.vs
    print(m.shape, s.shape, v.shape, v.max())

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    h = ax.contourf(m, s, v, 20, cmap='viridis', vmin=0, vmax=1.0)
    cbar = fig.colorbar(h)
    cbar.set_label(r"$P$")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma$")
    fig.savefig(os.path.abspath(dir_name + "/output/weights.png"), bbox_inches="tight", dpi=300)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    dir_name = os.path.dirname(__file__)
    t = os.path.abspath(dir_name + "/output/data_%s")
    plot_file = os.path.abspath(dir_name + "/output/surfaces.png")
    walk_file = os.path.abspath(dir_name + "/output/walk_%s.png")

    model_un = EfficiencyModelUncorrected(np.random.random(size=(10, 7)), np.random.random(10))
    pgm_file = os.path.abspath(dir_name + "/output/pgm.png")
    # fig = model_un.get_pgm(pgm_file)
    # plot_weights(dir_name)
    c = ChainConsumer()
    v = Viewer([[100, 300], [0, 70]], parameters=[r"$\mu$", r"$\sigma$"], truth=[200, 40])
    n = 2
    w = 8
    colours = ["#4CAF50", "#D32F2F", "#1E88E5"] * n

    for i in range(n):
        mean, std, threshold, lall, zall, fall, mask, num_obs = get_data(seed=i, n=500)
        theta = [mean, std]

        kwargs = {"num_steps": 12000, "num_burn": 4000, "save_interval": 60,
                  "plot_covariance": True, "unify_latent": True}  # , "callback": v.callback
        sampler = BatchMetroploisHastings(num_walkers=w, kwargs=kwargs, temp_dir=t % i, num_cores=4)

        model_good = EfficiencyModelUncorrected(fall, zall, name="Good%d" % i)
        model_good.fit(sampler, chain_consumer=c)

        model_un = EfficiencyModelUncorrected(fall[mask], zall[mask], name="Uncorrected%d" % i)
        model_un.fit(sampler, chain_consumer=c)

        model_cor = EfficiencyModelCorrected(fall[mask], zall[mask], threshold, num_obs,
                                             dir_name + "/output", name="Corrected%d" % i)
        model_cor.fit(sampler, chain_consumer=c)

    c.configure_bar(shade=True)
    c.configure_general(bins=1.0, colours=colours)
    c.configure_contour(sigmas=[0, 0.01, 1, 2], contourf=True, contourf_alpha=0.3)
    c.plot(filename=plot_file, truth=theta, figsize=(5, 5), legend=False)
    for i in range(len(c.chains)):
        c.plot_walks(filename=walk_file % c.names[i], chain=i, truth=theta)
        c.divide_chain(i, w).configure_general(rainbow=True) \
            .plot(figsize=(5, 5), filename=plot_file.replace(".png", "_%s.png" % c.names[i]),
                  truth=theta)
