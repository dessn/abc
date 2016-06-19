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
        super().__init__("c_o", r"$\mathbf{\hat{c}_i}$", data, group="Obs. Counts")


class ObservedRedshift(ParameterObserved):
    def __init__(self, data):
        super().__init__("z_o", "$\hat{z}$", data, group="Obs. Redshift")


class ObservedTimes(ParameterObserved):
    def __init__(self, data):
        super().__init__("t_o", r"$\mathbf{\hat{t}}$", data, group="Obs. Time")


class PeakLuminosity(ParameterLatent):
    def __init__(self, n, estimated_zeros, estimated_redshift, lum):
        super().__init__("L", "$L0$", n, group="Peak Luminosity")
        self.estz = estimated_zeros
        self.estr = estimated_redshift
        self.lum = lum

    def get_suggestion_requirements(self):
        return ["c_o", "z_o"]

    def get_suggestion(self, data):
        # flux = data["c_o"] / np.power(10, self.estz / 2.5)
        # max_flux = flux.max(axis=2).max(axis=1)
        # luminosity = max_flux / (data["z_o"] ** 2)
        # return luminosity
        return self.lum

    def get_suggestion_sigma(self, data):
        # flux = data["c_o"] / np.power(10, self.estz / 2.5)
        # max_flux = flux.max(axis=2).max(axis=1)
        # luminosity = max_flux / (data["z_o"] ** 2)
        # return 0.5 * luminosity
        return 0.001 * self.lum


class PeakTime(ParameterLatent):
    def __init__(self, n, ts):
        super().__init__("t", r"$t0$", n, group="Peak Time")
        self.ts = ts

    def get_suggestion_requirements(self):
        return ["c_o", "t_o"]

    def get_suggestion(self, data):
        # max_counts = data["c_o"].max(axis=2)
        # index2 = max_counts.argmax(axis=1)
        # index1 = np.arange(index2.size)
        # ts = data["t_o"]
        # tmax = ts[index1, index2]
        # return tmax
        return self.ts

    def get_suggestion_sigma(self, data):
        # return 10 * np.ones(self.num_param)
        return 0.01 * self.ts


class Stretch(ParameterLatent):
    def __init__(self, n, ss):
        super().__init__("s", "$s$", n, group="Stretch")
        self.ss = ss

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return self.ss

    def get_suggestion_sigma(self, data):
        return 0.01 * self.ss


class Luminosity(ParameterTransformation):
    def __init__(self):
        super().__init__("lum", "$L$", group="Luminosity")


class Flux(ParameterTransformation):
    def __init__(self):
        super().__init__("f", "$f$", group="Flux")


class Counts(ParameterTransformation):
    def __init__(self):
        super().__init__("c_i", "$c_i$", group="Counts")


class LuminosityMean(ParameterUnderlying):
    def __init__(self):
        super().__init__("mu_L", r"$\mu_L$", group="Luminosity Dist.")

    def get_log_prior(self, data):
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 500

    def get_suggestion_sigma(self, data):
        return 10


class LuminositySigma(ParameterUnderlying):
    def __init__(self):
        super().__init__("sigma_L", r"$\sigma_L$", group="Luminosity Dist.")

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 50

    def get_suggestion_sigma(self, data):
        return 2

    def get_log_prior(self, data):
        if data["sigma_L"] < 0:
            return -np.inf
        return 1


class StretchMean(ParameterUnderlying):
    def __init__(self):
        super().__init__("mu_s", r"$\mu_s$", group="Stretch Dist.")

    def get_log_prior(self, data):
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 10

    def get_suggestion_sigma(self, data):
        return 3


class StretchSigma(ParameterUnderlying):
    def __init__(self):
        super().__init__("sigma_s", r"$\sigma_s$", group="Stretch Dist.")

    def get_log_prior(self, data):
        if data["sigma_s"] < 0:
            return -np.inf
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 2

    def get_suggestion_sigma(self, data):
        return 1


class Calibration(ParameterUnderlying):
    def __init__(self, z, cov):
        super().__init__("zero", "${Z_i}$", num_param=z.size, group="Calibration")
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
        result = np.dot(flux[:, :, None], factor[None, :])
        return {"c_i": result}


class ToFlux(EdgeTransformation):
    def __init__(self):
        super().__init__("f", ["lum", "z_o"])

    def get_transformation(self, data):
        lum = data["lum"]
        z = data["z_o"][:, None]
        return {"f": lum / (z * z)}


class ToLuminosity(EdgeTransformation):
    def __init__(self):
        super().__init__("lum", ["L", "t", "t_o", "s"])

    def get_transformation(self, data):
        l0 = data["L"][:, None]
        s = data["s"][:, None]
        diff = data["t"][:, None] - data["t_o"]
        vals = np.exp(-(diff * diff) / (2 * s * s))
        return {"lum": l0 * vals}


class ToObservedCounts(Edge):
    def __init__(self):
        super().__init__("c_o", "c_i")
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        diff = data["c_o"] - data["c_i"]
        sigma = np.sqrt(data["c_i"])
        res = -(diff * diff) / (2 * sigma * sigma) - self.factor - np.log(sigma)
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
                 observed_zero_points, actual_lum, actual_stretch, actual_t0, seed=0, name="Uncorrected"):
        super().__init__(name)
        np.random.seed(seed)
        # Observed Parameters
        self.add(ObservedRedshift(observed_redshift))
        self.add(ObservedCounts(observed_counts))
        self.add(ObservedTimes(observed_times))

        # Latent Parameters
        self.add(PeakLuminosity(observed_redshift.size, observed_zero_points, observed_redshift, actual_lum))
        self.add(PeakTime(observed_redshift.size, actual_t0))
        self.add(Stretch(observed_redshift.size, actual_stretch))

        # Transformed Parameters
        self.add(Luminosity())
        self.add(Flux())
        self.add(Counts())

        # Underlying parameters
        self.add(LuminosityMean())
        self.add(LuminositySigma())
        self.add(StretchMean())
        self.add(StretchSigma())
        self.add(Calibration(observed_zero_points, observed_calibration))

        # Transformed Edges
        self.add(ToLuminosity())
        self.add(ToCounts())
        self.add(ToFlux())

        # Edges
        self.add(ToLuminosityDistribution())
        self.add(ToStretchDistribution())
        self.add(ToObservedCounts())

        self.finalise()


def get_weights(mul, sigmal, mus, sigmas, z0, z1, threshold, n=1e4):
    n = int(n)
    ls = np.random.normal(loc=mul, scale=sigmal, size=n)
    ss = np.random.normal(loc=mus, scale=sigmas, size=n)
    zs = np.random.uniform(1.1, high=1.5, size=n)

    ts = np.arange(-30, 30, 1)
    interior = -(ts * ts)[:, None] / (2 * ss * ss)[None, :]
    fluxes = (ls / (zs * zs))[None, :] * np.exp(interior)

    masks = []
    for z in [z0, z1]:
        factor = np.power(10, z / 2.5)
        counts = fluxes * factor
        zm = counts <= 0
        counts[zm] = 0.1
        realised_counts = np.random.normal(loc=counts, scale=np.sqrt(counts), size=counts.shape)
        mask = realised_counts > threshold
        masks.append(mask.sum(axis=0) > 1)
    masks = np.array(masks)
    final_mask = masks.sum(axis=0) > 0
    mean = final_mask.mean()
    return mean


def get_data(seed=5, n=30):
    np.random.seed(seed=seed)

    # Experimental Configuration
    num_obs = 60
    t_sep = 1
    threshold = 300

    # Zero points
    zeros = np.array([0.0, 0.0])
    calibration = 0.001 * np.identity(zeros.size)
    factor = np.power(10, zeros / 2.5)

    # Distributions
    lmu = 500
    lsigma = 50
    smu = 10
    ssigma = 2

    # Data
    zs = []
    ts = []
    ls = []
    cs = []
    mask = []
    ss = []
    t0s = []
    for i in range(n):
        t0 = np.random.randint(low=1, high=300)
        t = np.arange(-t_sep * (num_obs // 2), t_sep * (num_obs // 2), t_sep)
        l0 = np.random.normal(loc=lmu, scale=lsigma)
        s = np.random.normal(loc=smu, scale=ssigma)
        z = np.random.uniform(1.1, 1.5)

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
        ss.append(s)
        t0s.append(t0)
    zs = np.array(zs)
    ts = np.array(ts)
    cs = np.array(cs)
    ss = np.array(ss)
    ls = np.array(ls)
    t0s = np.array(t0s)
    mask = np.array(mask)
    print(mask.sum(), n, mask.sum()/n, ls.mean(), ls[mask].mean(), np.std(ls), np.std(ls[mask]))

    return lmu, lsigma, smu, ssigma, zeros, calibration, threshold, ls, ss, t0s, zs, ts, cs, mask, num_obs


def plot_data(dir_name):
    lmu, lsigma, smu, ssigma, zeros, calibration, threshold, ls, ss, t0s, zs, ts, cs, mask, num_obs = get_data()

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

    # plot_data(dir_name)
    # lmu, lsigma, smu, ssigma, zeros, calibration, threshold, ls, ss, t0s, zs, ts, cs, mask, num_obs = get_data(n=1000)
    # model = EfficiencyModelUncorrected(cs, zs, ts, calibration, zeros)
    # pgm_file = os.path.abspath(dir_name + "/output/pgm.png")
    # fig = model.get_pgm(pgm_file, seed=3)
    c = ChainConsumer()
    n = 1
    w = 4
    colours = ["#4CAF50", "#D32F2F", "#1E88E5"] * n
    for i in range(n):
        lmu, lsigma, smu, ssigma, zeros, calibration, threshold, ls, ss, t0s, zs, ts, cs, mask, num_obs = get_data()
        theta = [lmu, lsigma, smu, ssigma] + zeros.tolist()
        # theta = [lmu, lsigma] + zeros.tolist()
        theta2 = theta + ls.tolist() + t0s.tolist() + ss.tolist()

        kwargs = {"num_steps": 2000, "num_burn": 310000, "save_interval": 60,
                  "plot_covariance": True}
        sampler = BatchMetroploisHastings(num_walkers=w, kwargs=kwargs, temp_dir=t % i, num_cores=4)

        model_good = EfficiencyModelUncorrected(cs, zs, ts, calibration, zeros, ls, ss, t0s, name="Good%d" % i)
        model_good.fit(sampler, chain_consumer=c)

        model_un = EfficiencyModelUncorrected(cs[mask], zs[mask], ts[mask], calibration,
                                              zeros, ls[mask], ss[mask], t0s[mask], name="Uncorrected%d" % i)
        model_un.fit(sampler, chain_consumer=c)

        biased_chain = c.chains[-1]
        # model_cor.fit(sampler, chain_consumer=c)

        filename = dir_name + "/output/weights.txt"
        if not os.path.exists(filename):
            weights = []
            for i, row in enumerate(biased_chain):
                weights.append(get_weights(row[0], row[1], row[2], row[3], row[4], row[5], threshold))
                print(100.0 * i / biased_chain.shape[0])
            weights = np.array(weights)
            np.savetxt(filename, weights)
        else:
            weights = np.power(np.loadtxt(filename), mask.sum())
        c.add_chain(biased_chain, name="Importance Sampled", weights=1/weights)

    c.configure_bar(shade=True)
    c.configure_general(bins=1.0, colours=colours)
    c.configure_contour(sigmas=[0, 0.01, 1, 2], contourf=True, contourf_alpha=0.2)
    c.plot(filename=plot_file, truth=theta, figsize=(5, 5), legend=False, parameters=6)
    for i in range(len(c.chains)):
        c.plot_walks(filename=walk_file % c.names[i], chain=i, truth=theta)
        # c.divide_chain(i, w).configure_general(rainbow=True) \
        #     .plot(figsize=(5, 5), filename=plot_file.replace(".png", "_%s.png" % c.names[i]),
        #           truth=theta)
