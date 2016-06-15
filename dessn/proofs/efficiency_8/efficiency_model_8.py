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


class ObservedTimes(ParameterObserved):
    def __init__(self, data):
        super().__init__("t_o", r"$\mathbf{\hat{t}}$", data)


class Luminosity(ParameterLatent):
    def __init__(self, n, estimated_zeros, estimated_redshift):
        super().__init__("L", "$L$", n)
        self.estz = estimated_zeros
        self.estr = estimated_redshift

    def get_suggestion_requirements(self):
        return ["c_o", "z_o"]

    def get_suggestion(self, data):
        flux = data["c_o"] / np.power(10, self.estz / 2.5)
        luminosity = flux / (data["z_o"] ** 2)
        return np.max(luminosity)

    def get_suggestion_sigma(self, data):
        flux = data["c_o"] / np.power(10, self.estz / 2.5)
        luminosity = flux / (data["z_o"] ** 2)
        return 0.5 * (np.max(luminosity) - np.min(luminosity))


class PeakTime(ParameterLatent):
    def __init__(self, n):
        super().__init__("t", r"$t_0$", n)

    def get_suggestion_requirements(self):
        return ["c_o", "t_o"]

    def get_suggestion(self, data):
        return data["t_o"][data["c_o"].argmax()]

    def get_suggestion_sigma(self, data):
        return 10


class Stretch(ParameterLatent):
    def __init__(self, n):
        super().__init__("s", "$s$", n)

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 10

    def get_suggestion_sigma(self, data):
        return 7


class Flux(ParameterTransformation):
    def __init__(self):
        super().__init__("f", "$f$")


class Counts(ParameterTransformation):
    def __init__(self):
        super().__init__("c_i", "$c_i$")


class LuminosityMean(ParameterUnderlying):
    def __init__(self):
        super().__init__("mu_L", r"$\mu_L$")

    def get_log_prior(self, data):
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 500

    def get_suggestion_sigma(self, data):
        return 20


class LuminositySigma(ParameterUnderlying):
    def __init__(self):
        super().__init__("sigma_L", r"$\sigma_L$")

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


class StretchMean(ParameterUnderlying):
    def __init__(self):
        super().__init__("mu_s", r"$\mu_s$")

    def get_log_prior(self, data):
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 20

    def get_suggestion_sigma(self, data):
        return 3


class StretchSigma(ParameterUnderlying):
    def __init__(self):
        super().__init__("sigma_s", r"$\sigma_s$")

    def get_log_prior(self, data):
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 5

    def get_suggestion_sigma(self, data):
        return 3


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


class ToFlux(EdgeTransformation):
    def __init__(self):
        super().__init__("f", ["L", "t", "s", "t_o", "z_o"])

    def get_transformation(self, data):
        l0 = data["L"]
        s = data["s"]
        diff = data["t"] - data["t_o"]
        z = data["z_o"]
        return l0 * np.exp(-(diff * diff) / (2 * s * s)) / (z * z)


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


class ToLuminosityDistribution(Edge):
    def __init__(self):
        super().__init__("L", ["mu_L", "sigma_L"])
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["L"] - data["mu_L"]
        sigma = data["sigma_L"]
        return -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)


class ToStretchDistribution(Edge):
    def __init__(self):
        super().__init__("s", ["mu_s", "sigma_s"])
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["s"] - data["mu_s"]
        sigma = data["sigma_s"]
        return -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)


class EfficiencyModelUncorrected(Model):
    def __init__(self, observed_counts, observed_redshift, observed_times, observed_calibration,
                 observed_zero_points, seed=0, name="Uncorrected"):
        super().__init__(name)
        np.random.seed(seed)
        # Observed Parameters
        self.add(ObservedRedshift(observed_redshift))
        self.add(ObservedCounts(observed_counts))
        self.add(ObservedTimes(observed_times))

        # Latent Parameters
        self.add(Luminosity(observed_redshift.size, observed_zero_points, observed_redshift))
        self.add(PeakTime(observed_redshift.size))
        self.add(Stretch(observed_redshift.size))

        # Transformed Parameters
        self.add(Counts())
        self.add(Flux())

        # Underlying parameters
        self.add(LuminosityMean())
        self.add(LuminositySigma())
        self.add(StretchMean())
        self.add(StretchSigma())
        self.add(Calibration(observed_zero_points, observed_calibration))

        # Transformed Edges
        self.add(ToCounts())
        self.add(ToFlux())

        # Edges
        self.add(ToLuminosityDistribution())
        self.add(ToStretchDistribution())
        self.add(ToObservedCounts())

        self.finalise()


def get_data(seed=5, n=400):
    np.random.seed(seed=seed)

    # Experimental Configuration
    num_obs = 20
    t_sep = 2
    threshold = 300

    # Zero points
    zeros = np.array([0.0, 0.0])
    calibration = 0.01 * np.identity(zeros.size)
    factor = np.power(10, zeros / 2.5)

    # Distributions
    lmu = 500
    lsigma = 50
    smu = 20
    ssigma = 4

    # Data
    zs = []
    ts = []
    ls = []
    cs = []
    mask = []
    for i in range(n):
        t0 = np.random.randint(low=1, high=300)
        t = np.arange(-t_sep * (num_obs // 2), t_sep * (num_obs // 2), t_sep)
        l0 = np.random.normal(loc=lmu, scale=lsigma)
        s = np.random.normal(loc=smu, scale=ssigma)
        z = np.random.uniform(0.5, 1.5)


        lums = l0 * np.exp(-(t * t) / (2 * s * s))
        flux = lums / (z * z)
        counts = np.dot(flux[:, None], factor[None, :])

        c_o = np.array([np.random.normal(scale=np.sqrt(a)) for a in counts.flatten()]).reshape(counts.shape)
        c_o += counts
        mask_point = c_o > threshold
        mask_band = mask_point.sum(axis=1) >= 2
        mask_all = mask_band.sum() > 0

        zs.append(z)
        ts.append(t + t0)
        cs.append(c_o)
        mask.append(mask_all)
        ls.append(l0)

    zs = np.array(zs)
    ts = np.array(ts)
    cs = np.array(cs)
    ls = np.array(ls)
    mask = np.array(mask)

    print(mask.sum(), n, mask.sum()/n, ls.mean(), ls[mask].mean(), np.std(ls), np.std(ls[mask]))

    return lmu, lsigma, smu, ssigma, zeros, calibration, threshold, ls, zs, ts, cs, mask, num_obs


def plot_data(dir_name):
    lmu, lsigma, smu, ssigma, zeros, calibration, threshold, ls, zs, ts, cs, mask, num_obs = get_data()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel("$z$")
    ax.set_ylabel("$L$")
    ax.scatter(zs[mask], ls[mask], color="#1976D2", alpha=0.7, label="Discovered")
    ax.scatter(zs[~mask], ls[~mask], color="#D32F2F", alpha=0.7, label="Undiscovered")
    ax.axhline(lmu, color="k", ls="--")
    ax.legend(loc=3, fontsize=12)
    fig.savefig(os.path.abspath(dir_name + "/output/data.png"), bbox_inces="tight", dpi=300)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    dir_name = os.path.dirname(__file__)
    t = os.path.abspath(dir_name + "/output/data_%s")
    plot_file = os.path.abspath(dir_name + "/output/surfaces.png")
    walk_file = os.path.abspath(dir_name + "/output/walk_%s.png")

    plot_data(dir_name)
    lmu, lsigma, smu, ssigma, zeros, calibration, threshold, ls, zs, ts, cs, mask, num_obs = get_data()
    model = EfficiencyModelUncorrected(cs, zs, ts, calibration, zeros)
    pgm_file = os.path.abspath(dir_name + "/output/pgm.png")
    fig = model.get_pgm(pgm_file)

    exit()
    c = ChainConsumer()
    v = Viewer([[100, 300], [0, 70]], parameters=[r"$\mu$", r"$\sigma$"], truth=[200, 40])
    n = 3
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

        model_cor.fit(sampler, chain_consumer=c)

    c.configure_bar(shade=True)
    c.configure_general(bins=1.0, colours=colours)
    c.configure_contour(sigmas=[0, 0.01, 1, 2], contourf=True, contourf_alpha=0.2)
    c.plot(filename=plot_file, truth=theta, figsize=(5, 5), legend=False, parameters=2)
    for i in range(len(c.chains)):
        c.plot_walks(filename=walk_file % c.names[i], chain=i, truth=theta)
        # c.divide_chain(i, w).configure_general(rainbow=True) \
        #     .plot(figsize=(5, 5), filename=plot_file.replace(".png", "_%s.png" % c.names[i]),
        #           truth=theta)
