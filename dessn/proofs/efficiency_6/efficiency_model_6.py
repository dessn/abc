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
from scipy.interpolate import RegularGridInterpolator, interp1d


class ObservedCounts(ParameterObserved):
    def __init__(self, data):
        super().__init__("c_o", r"$\mathbf{\hat{c}_i}$", data)


class ObservedRedshift(ParameterObserved):
    def __init__(self, data):
        super().__init__("z_o", "$\hat{z}$", data)


class LatentFlux(ParameterLatent):
    def __init__(self, n, zeros):
        super().__init__("f", "$f$", n)
        self.zeros = zeros

    def get_suggestion_requirements(self):
        return ["c_o"]

    def get_suggestion(self, data):
        factor = (1 / np.power(10, self.zeros / 2.5))[None, :, None]
        res = data["c_o"] * factor
        res = np.mean(np.mean(res, axis=2), axis=1)
        return res

    def get_suggestion_sigma(self, data):
        factor = (1 / np.power(10, self.zeros / 2.5))[None, :, None]
        res = data["c_o"] * factor
        res = np.std(np.mean(res, axis=1), axis=1)
        return res


class Luminosity(ParameterTransformation):
    def __init__(self):
        super().__init__("L", "$L$")


class Counts(ParameterTransformation):
    def __init__(self):
        super().__init__("c_i", "$c_i$")


class Mean(ParameterUnderlying):
    def __init__(self):
        super().__init__("mu", r"$\mu$")

    def get_log_prior(self, data):
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 500

    def get_suggestion_sigma(self, data):
        return 20


class Scatter(ParameterUnderlying):
    def __init__(self):
        super().__init__("sigma", r"$\sigma$")

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 50

    def get_suggestion_sigma(self, data):
        return 10

    def get_log_prior(self, data):
        if data["sigma"] < 0:
            return -np.inf
        return 1


class Calibration(ParameterUnderlying):
    def __init__(self, z, cov):
        super().__init__("zero", "${Z_i}$", num_param=z.size)
        self.z = z
        self.icov = np.linalg.inv(cov)
        self.diag = np.diag(cov)

    def get_suggestion(self, data):
        return self.z

    def get_suggestion_sigma(self, data):
        return self.diag

    def get_log_prior(self, data):
        diff = data["zero"] - self.z
        chi2 = -0.5 * np.dot(diff[None, :], np.dot(self.icov, diff[:, None]))
        return chi2


class ToCounts(EdgeTransformation):
    def __init__(self):
        super().__init__("c_i", ["f", "zero"])

    def get_transformation(self, data):
        flux = data["f"]
        z = data["zero"]
        factor = np.power(10, z / 2.5)
        result = np.dot(flux[:, None], factor[None, :])
        return {"c_i": result}


class ToObservedCounts(Edge):
    def __init__(self):
        super().__init__("c_o", "c_i")
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["c_o"] - data["c_i"][:, :, None]
        sigma = np.sqrt(data["c_i"])
        sigma = sigma * sigma
        sigma = sigma[:, :, None]
        res = -0.5 * diff * diff / sigma - self.factor - np.log(sigma)
        res[np.isnan(res)] = -np.inf
        res = res.sum(axis=1).sum(axis=1)
        return res


class ToLuminosity(EdgeTransformation):
    def __init__(self):
        super().__init__("L", ["f", "z_o"])

    def get_transformation(self, data):
        return {"L": data["f"] * data["z_o"] * data["z_o"]}, 2 * np.log(data["z_o"])


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
        super().__init__(["z_o"], ["mu", "sigma", "zero"])
        self.filename = temp_dir + os.sep + "bias_correction_%d.npy" % threshold
        self.threshold = threshold

        self.mus = None
        self.sigmas = None
        self.z0s = None
        self.z1s = None
        self.vs = None

        self.zs = np.linspace(0.5, 1.5, 50)
        self.cs = np.linspace(1, 3000, 1000)
        self.fs = np.linspace(10, 2500, 1000)

        bound = self.cs - self.threshold
        ltz = (bound > 0) * 2.0 - 1.0
        self.gplus = ltz * 0.5 * erf((np.abs(bound)) / (np.sqrt(2 * self.cs))) + 0.5
        self.gminus = 1 - self.gplus
        self.gminusN = np.power(self.gminus, number)
        self.gminusNmo = np.power(self.gminus, number - 1)
        self.N = number

        self.x1 = self.gminusN
        self.x2 = self.N * self.gplus * self.gminusNmo

        self.interp = self.get_data()

    # def _get_s0(self, mu, sigma, Z):
    #     fs = self.cs / np.power(10, Z / 2.5)
    #     zvals = np.zeros(self.zs.shape)
    #     for i, z in enumerate(self.zs):
    #         lum = fs * (z * z)
    #         diff = lum - mu
    #         gauss = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(diff * diff) / (2 * sigma * sigma))
    #         zvals[i] = simps(gauss * self.gminusN, x=fs)
    #     return simps(self.zs * self.zs * zvals, x=self.zs)
    #
    # def _get_s1(self, mu, sigma, Z):
    #     fs = self.cs / np.power(10, Z / 2.5)
    #     zvals = np.zeros(self.zs.shape)
    #     for i, z in enumerate(self.zs):
    #         diff = fs * (z * z) - mu
    #         gauss = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(diff * diff) / (2 * sigma * sigma))
    #         integral = simps(gauss * self.N * self.gplus * self.gminusNmo, x=fs)
    #         # if mu == 500 and sigma == 50 and i in [0, self.zs.size - 1]:
    #         #     # print(fs, lum, diff)
    #         #     plt.plot(fs, 20 * gauss, lw=3)
    #         #     print("Area is ", z * z * simps(gauss, x=fs), " at redshift %f " % z)
    #         #     plt.plot(fs, self.N * self.gplus * self.gminusNmo)
    #         #     plt.title("%f %f %f" % (mu, sigma, z))
    #         #     plt.show()
    #         zvals[i] = integral
    #     if mu == 500 and sigma == 50:
    #         plt.plot(self.zs, zvals)
    #         plt.plot(self.zs, self.zs * self.zs * zvals)
    #         plt.show()
    #         print("VAL IS ", simps(self.zs * self.zs * zvals, x=self.zs))
    #     return simps(self.zs * self.zs * zvals, x=self.zs)

    def _get_val(self, mu, sigma, z1, z2):
        zvals = np.zeros(self.zs.shape)
        f1 = self.cs / np.power(10, z1 / 2.5)
        f2 = self.cs / np.power(10, z2 / 2.5)

        # print(f1, self.fs)
        vals11 = interp1d(f1, self.x1)(self.fs)
        vals12 = interp1d(f1, self.x2)(self.fs)
        vals21 = interp1d(f2, self.x1)(self.fs)
        vals22 = interp1d(f2, self.x2)(self.fs)

        summed = vals11 * vals21 + vals11 * vals22 + vals12 * vals21 + vals12 * vals22

        for i, z in enumerate(self.zs):
            diff = self.fs * z * z - mu
            gauss = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(diff * diff) / (2 * sigma * sigma))
            integral = simps(gauss * summed, x=self.fs)
            zvals[i] = integral
        #     if mu == 500.0 and sigma == 50.0 and (i == 0 or i == self.zs.size - 1):
        #         plt.plot(self.fs, summed)
        #         plt.plot(self.fs, gauss * 100, lw=2)
        #         plt.show()
        # if mu == 500.0 and sigma == 50.0:
        #     plt.plot(self.zs, zvals)
        #     plt.show()
        return simps(self.zs * self.zs * zvals, x=self.zs)

    def get_data(self):
        self.mus = np.linspace(400, 600, 30)
        self.sigmas = np.linspace(30, 70, 15)
        self.z0s = np.linspace(-0.01, 0.01, 7)
        self.z1s = np.linspace(-0.01, 0.01, 7)

        if os.path.exists(self.filename):
            self.vs = np.load(self.filename)
        else:
            self.vs = np.zeros((self.mus.size, self.sigmas.size, self.z0s.size, self.z1s.size))
            for i, m in enumerate(self.mus):
                print("Starting mu: %d/%d" % (i+1, self.mus.size))
                for j, s in enumerate(self.sigmas):
                    for k, z0 in enumerate(self.z0s):
                        for l, z1 in enumerate(self.z1s):
                            # Please kill me
                            res = 1 - self._get_val(m, s, z0, z1)
                            self.vs[i, j, k, l] = res

            np.save(self.filename, self.vs)
        return RegularGridInterpolator((self.mus, self.sigmas, self.z0s, self.z1s), self.vs,
                                       bounds_error=False, fill_value=0.0)

    def get_log_likelihood(self, data):
        mu = data["mu"]
        sigma = data["sigma"]
        zeros = data["zero"]
        redshift = data["z_o"]

        points = [[mu, sigma, zeros[0], zeros[1]]]
        p = self.interp(points)
        ps = -np.log(p)
        ps[ps == np.inf] = -np.inf
        return np.ones(redshift.shape) * ps


class EfficiencyModelUncorrected(Model):
    def __init__(self, observed_counts, observed_redshift, observed_calibration,
                 observed_zero_points, seed=0, name="Uncorrected"):
        super().__init__(name)
        np.random.seed(seed)
        self.add(ObservedRedshift(observed_redshift))
        self.add(ObservedCounts(observed_counts))

        self.add(LatentFlux(observed_redshift.size, observed_zero_points))
        self.add(Counts())
        self.add(Mean())
        self.add(Scatter())
        self.add(Calibration(observed_zero_points, observed_calibration))
        self.add(Luminosity())

        self.add(ToCounts())
        self.add(ToLuminosity())
        self.add(ToUnderlying())
        self.add(ToObservedCounts())
        self.finalise()


class EfficiencyModelCorrected(EfficiencyModelUncorrected):
    def __init__(self, observed_flux, observed_redshift, observed_calibration,
                 observed_zero_points, threshold, number, temp_dir, seed=0, name="Corrected"):
        super().__init__(observed_flux, observed_redshift, observed_calibration,
                 observed_zero_points, seed=seed, name=name)

        self.add(BiasCorrection(threshold, number, temp_dir))
        self.finalise()


def get_data(seed=5, n=400):
    np.random.seed(seed=seed+4)
    num_obs = 2
    mean = 500.0
    std = 50.0
    zeros = np.array([0.0, 0.0])
    calibration = 0.01 * np.identity(zeros.size)

    z_start = 0.5
    z_end = 1.5
    threshold = 300
    factor = np.power(10, zeros / 2.5)
    z_o = np.random.uniform(z_start, z_end, size=n)
    lum = np.random.normal(loc=mean, scale=std, size=n)
    flux = lum / (z_o * z_o)

    ac = np.dot(flux[:, None], factor[None, :])
    c_o = np.dstack((ac + np.random.normal(scale=np.sqrt(ac), size=ac.shape) for i in range(num_obs)))
    obs_mask = c_o > threshold
    mask = obs_mask.sum(axis=2) >= 1
    mask = mask.sum(axis=1) > 0
    print(mask.sum(), n, mask.sum()/n, lum.mean(), lum[mask].mean(), np.std(lum), np.std(lum[mask]))

    return mean, std, zeros, calibration, threshold, lum, z_o, c_o, mask, num_obs


def plot_weights(dir_name):
    mean, std, zeros, calibration, threshold, lall, zall, call, mask, num_obs = get_data()

    bias = BiasCorrection(threshold, num_obs, dir_name + "/output")
    m, s, z0, z1 = np.meshgrid(bias.mus, bias.sigmas, bias.z0s, bias.z1s, indexing='ij')
    v = bias.vs
    print(m.shape, s.shape, v.shape, v.max())
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    n1 = m.shape[2] // 2
    n2 = m.shape[3] // 2
    h = ax.contourf(m[:,:,n1,n2], s[:,:,n1,n2], v[:,:,n1,n2], 20, cmap='viridis')  #, vmin=0, vmax=1.0
    cbar = fig.colorbar(h)
    cbar.set_label(r"$P$")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma$")
    fig.savefig(os.path.abspath(dir_name + "/output/weights.png"), bbox_inches="tight", dpi=300)


def plot_data(dir_name):
    mean, std, zeros, calibration, threshold, lall, zall, call, mask, num_obs = get_data()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel("$z$")
    ax.set_ylabel("$L$")
    ax.scatter(zall[mask], lall[mask], color="#1976D2", alpha=0.7, label="Discovered")
    ax.scatter(zall[~mask], lall[~mask], color="#D32F2F", alpha=0.7, label="Undiscovered")
    ax.axhline(mean, color="k", ls="--")
    ax.legend(loc=3, fontsize=12)
    fig.savefig(os.path.abspath(dir_name + "/output/data.png"), bbox_inces="tight", dpi=300)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    dir_name = os.path.dirname(__file__)
    t = os.path.abspath(dir_name + "/output/data_%s")
    plot_file = os.path.abspath(dir_name + "/output/surfaces.png")
    walk_file = os.path.abspath(dir_name + "/output/walk_%s.png")

    # plot_data(dir_name)
    mean, std, zeros, calibration, threshold, lall, zall, call, mask, num_obs = get_data()
    # model_un = EfficiencyModelUncorrected(call, zall, calibration, zeros)
    # pgm_file = os.path.abspath(dir_name + "/output/pgm.png")
    # fig = model_un.get_pgm(pgm_file)
    plot_weights(dir_name)
    c = ChainConsumer()
    v = Viewer([[100, 300], [0, 70]], parameters=[r"$\mu$", r"$\sigma$"], truth=[200, 40])
    n = 1
    w = 4
    colours = ["#4CAF50", "#D32F2F", "#1E88E5"] * n
    for i in range(n):
        mean, std, zeros, calibration, threshold, lall, zall, call, mask, num_obs = get_data()
        theta = [mean, std] + zeros.tolist()

        kwargs = {"num_steps": 40000, "num_burn": 30000, "save_interval": 60,
                  "plot_covariance": True}  # , "unify_latent": True # , "callback": v.callback
        sampler = BatchMetroploisHastings(num_walkers=w, kwargs=kwargs, temp_dir=t % i, num_cores=4)

        model_good = EfficiencyModelUncorrected(call, zall, calibration, zeros, name="Good%d" % i)
        model_good.fit(sampler, chain_consumer=c)  # , include_latent=True

        model_un = EfficiencyModelUncorrected(call[mask], zall[mask], calibration,
                                              zeros, name="Uncorrected%d" % i)
        model_un.fit(sampler, chain_consumer=c)

        model_cor = EfficiencyModelCorrected(call[mask], zall[mask], calibration, zeros,
                                             threshold, num_obs,
                                             dir_name + "/output", name="Corrected%d" % i)
        model_cor.fit(sampler, chain_consumer=c)

    c.configure_bar(shade=True)
    c.configure_general(bins=1.0, colours=colours)
    c.configure_contour(sigmas=[0, 0.01, 1, 2], contourf=True, contourf_alpha=0.2)
    c.plot(filename=plot_file, truth=theta, figsize=(5, 5), legend=False)
    for i in range(len(c.chains)):
        c.plot_walks(filename=walk_file % c.names[i], chain=i, truth=theta)
        # c.divide_chain(i, w).configure_general(rainbow=True) \
        #     .plot(figsize=(5, 5), filename=plot_file.replace(".png", "_%s.png" % c.names[i]),
        #           truth=theta)
