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
        super().__init__("f_o", "$f_o$", data)


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
        return data["f_o"]

    def get_suggestion_sigma(self, data):
        return 0.5 * data["f_o"]


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
        s = data["sigma"]
        return -2 * np.log(s)


class ToLatentFlux(Edge):
    def __init__(self):
        super().__init__(["f_o"], "f")
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["f_o"] - data["f"]
        sigma = np.sqrt(data["f"])
        res = -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)
        res[np.isnan(res)] = -np.inf
        return res


class ToLuminosityCorrection(Edge):
    def __init__(self):
        super().__init__("L", "z_o")

    def get_log_likelihood(self, data):
        return np.log(1 + data["z_o"])


class ToLuminosity(EdgeTransformation):
    def __init__(self):
        super().__init__("L", ["f", "z_o"])

    def get_transformation(self, data):
        return {"L": data["f"] * (1 + data["z_o"])}


class ToUnderlying(Edge):
    def __init__(self):
        super().__init__("L", ["mu", "sigma"])
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["L"] - data["mu"]
        sigma = data["sigma"]
        return -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)


class BiasCorrection(Edge):
    def __init__(self, threshold, temp_dir):
        super().__init__(["z_o"], ["mu", "sigma"])
        self.filename = temp_dir + os.sep + "bias_correction.npy"
        self.threshold = threshold
        self.mus = None
        self.sigmas = None
        self.zs = None
        self.vs = None
        self.ms = None
        self.ss = None
        self.z = None
        self.interp = self.get_data()

    def get_data(self):
        self.mus = np.linspace(100, 300, 40)
        self.sigmas = np.linspace(5, 100, 40)
        self.zs = np.linspace(0.4, 1.6, 40)
        ms, ss, zs = np.meshgrid(self.mus, self.sigmas, self.zs)
        self.ms = ms
        self.ss = ss
        self.z = zs
        if os.path.exists(self.filename):
            self.vs = np.load(self.filename)
        else:
            fs = np.linspace(1, 500, 500)
            bound = fs - self.threshold
            ltz = (bound > 0) * 2.0 - 1.0
            erf_term = ltz * 0.5 * erf((np.abs(bound)) / (np.sqrt(2 * fs))) + 0.5
            vs = []
            for m, s, z in zip(ms.flatten(), ss.flatten(), zs.flatten()):
                # m = 50
                # s = 20
                # z = 1.5

                g = (fs * (1 + z) - m)
                gaussian = (1 + z) * (1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-g * g / (2 * s * s))
                integral = simps(gaussian * erf_term, x=fs)
                vs.append(integral)
                # print(m, s, z, integral, erf_term)
                # assert 1 == 2

            vs = np.array(vs)
            self.vs = vs.reshape((self.mus.size, self.sigmas.size, self.zs.size))
            np.save(self.filename, self.vs)
        return RegularGridInterpolator((self.mus, self.sigmas, self.zs), self.vs,
                                       bounds_error=False, fill_value=0.0)

    def get_log_likelihood(self, data):
        mu = data["mu"]
        sigma = data["sigma"]
        redshift = data["z_o"]

        points = [[mu, sigma, z] for z in redshift]
        ps = -np.log(self.interp(points))
        ps[ps == np.inf] = -np.inf
        return ps


class EfficiencyModelUncorrected(Model):
    def __init__(self, observed_flux, observed_redshift, seed=0, name="Uncorrected"):
        super().__init__(name)
        np.random.seed(seed)
        self.add(ObservedRedshift(observed_redshift))
        self.add(ObservedFlux(observed_flux))
        self.add(LatentFlux(observed_flux.size))
        self.add(ActualMean())
        self.add(ActualSigma())
        self.add(Luminosity())
        self.add(ToLuminosity())
        self.add(ToLatentFlux())
        self.add(ToUnderlying())
        self.add(ToLuminosityCorrection())
        self.finalise()


class EfficiencyModelCorrected(Model):
    def __init__(self, observed_flux, observed_redshift, threshold, temp_dir, seed=0, name="Corrected"):
        super().__init__(name)
        np.random.seed(seed)
        self.add(ObservedRedshift(observed_redshift))
        self.add(ObservedFlux(observed_flux))
        self.add(LatentFlux(observed_flux.size))
        self.add(ActualMean())
        self.add(ActualSigma())
        self.add(Luminosity())
        self.add(ToLatentFlux())
        self.add(ToLuminosity())
        self.add(BiasCorrection(threshold, temp_dir))
        self.add(ToUnderlying())
        self.add(ToLuminosityCorrection())
        self.finalise()


def get_data(seed=5, n=500):
    np.random.seed(seed=seed)
    mean = 200.0
    std = 40.0
    z_start = 0.45
    z_end = 0.5
    threshold = 36

    z_o = np.random.uniform(z_start, z_end, size=n)
    lum = np.random.normal(loc=mean, scale=std, size=n)
    flux = lum / (1 + z_o)
    f_o = flux + np.random.normal(scale=np.sqrt(flux), size=n)
    mask = f_o > threshold
    print(mask.sum(), n, lum[mask].mean())

    return mean, std, threshold, lum, z_o, f_o, mask


def plot_weights(dir_name):
    from mpl_toolkits.mplot3d import Axes3D
    mean, std, threshold, lall, zall, fall, mask = get_data(seed=0)

    E = BiasCorrection(threshold, dir_name + "/output")
    n = -15
    m = E.ms[::-1, ::-1, ::n].flatten()
    s = E.ss[::-1, ::-1, ::n].flatten()
    r = E.z[::-1, ::-1, ::n].flatten()
    z = E.vs[::-1, ::-1, ::n].flatten()

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    # h = ax.scatter(m, s, r, c=np.log(z), cmap='viridis', lw=0, vmin=0, vmax=1.0)
    h = ax.scatter(m, s, r, c=z, cmap='viridis', lw=0)
    cbar = fig.colorbar(h)
    cbar.set_label(r"$P$")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma$")
    ax.set_zlabel(r"$z$")
    fig.savefig(os.path.abspath(dir_name + "/output/weights.png"), bbox_inches="tight", dpi=300)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    dir_name = os.path.dirname(__file__)
    t = os.path.abspath(dir_name + "/output/data_%s")
    plot_file = os.path.abspath(dir_name + "/output/surfaces.png")
    walk_file = os.path.abspath(dir_name + "/output/walk_%s.png")

    model_un = EfficiencyModelUncorrected(np.random.random(10), np.random.random(10))
    pgm_file = os.path.abspath(dir_name + "/output/pgm.png")
    # fig = model_un.get_pgm(pgm_file)
    # plot_weights(dir_name)
    c = ChainConsumer()
    v = Viewer([[100, 300], [0, 70]], parameters=[r"$\mu$", r"$\sigma$"], truth=[200, 40])
    n = 1
    w = 4
    colours = ["#4CAF50", "#D32F2F", "#1E88E5"] * n

    for i in range(n):
        mean, std, threshold, lall, zall, fall, mask = get_data(seed=i)
        theta = [mean, std]
        #
        # plt.hist(lall, alpha=0.5, bins=20)
        # plt.hist(lall[mask], alpha=0.5, bins=20)
        # plt.show()
        #
        # plt.hist(zall, alpha=0.5, bins=20)
        # plt.hist(zall[mask], alpha=0.5, bins=20)
        # plt.show()
        #
        # plt.hist(fall, alpha=0.5, bins=20)
        # plt.hist(fall[mask], alpha=0.5, bins=20)
        # plt.show()
        # exit()

        kwargs = {"num_steps": 3000, "num_burn": 1000, "save_interval": 60,
                  "plot_covariance": True, "unify_latent": True}  # , "callback": v.callback
        sampler = BatchMetroploisHastings(num_walkers=w, kwargs=kwargs, temp_dir=t % i, num_cores=4)

        model_good = EfficiencyModelUncorrected(fall, zall, name="Good%d" % i)
        model_good.fit(sampler, chain_consumer=c)
        # print("Good ", model_good.get_log_posterior(theta_good), c.posteriors[-1][-1])

        # model_un = EfficiencyModelUncorrected(fall[mask], zall[mask], name="Uncorrected%d" % i)
        # model_un.fit(sampler, chain_consumer=c)
        # print("Uncorrected ", model_un.get_log_posterior(theta_bias), c.posteriors[-1][-1])

        model_cor = EfficiencyModelCorrected(fall[mask], zall[mask], threshold,
                                             dir_name + "/output", name="Corrected%d" % i)
        model_cor.fit(sampler, chain_consumer=c)
        # print("Corrected ", model_cor.get_log_posterior(theta_bias), c.posteriors[-1][-1])

    c.configure_bar(shade=True)
    c.configure_general(bins=1.0, colours=colours)
    c.configure_contour(sigmas=[0, 0.01, 1, 2], contourf=True, contourf_alpha=0.3)
    c.plot(filename=plot_file, truth=theta, figsize=(5, 5), legend=False)
    for i in range(len(c.chains)):
        c.plot_walks(filename=walk_file % c.names[i], chain=i, truth=theta)
        # c.divide_chain(i, w).configure_general(rainbow=True) \
        #     .plot(figsize=(5, 5), filename=plot_file.replace(".png", "_%s.png" % c.names[i]),
        #           truth=theta)
