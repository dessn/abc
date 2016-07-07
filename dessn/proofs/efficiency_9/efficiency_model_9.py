from joblib import Parallel, delayed
from dessn.framework.model import Model
from dessn.framework.edge import Edge, EdgeTransformation
from dessn.framework.parameter import ParameterObserved, ParameterLatent, ParameterUnderlying, \
    ParameterTransformation, ParameterDiscrete
from dessn.framework.samplers.batch import BatchMetropolisHastings
from dessn.chain.chain import ChainConsumer
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

from dessn.framework.samplers.ensemble import EnsembleSampler


class ObservedCounts(ParameterObserved):
    def __init__(self, data):
        super().__init__("c_o", r"$\mathbf{\hat{c}_i}$", data, group="Obs. Counts")


class ObservedRedshift(ParameterObserved):
    def __init__(self, data):
        super().__init__("z_o", "$\hat{z}$", data, group="Obs. Redshift")


class ObservedTimes(ParameterObserved):
    def __init__(self, data):
        super().__init__("t_o", r"$\mathbf{\hat{t}}$", data, group="Obs. Time")


class ObservedType(ParameterObserved):
    def __init__(self, data):
        super().__init__("type_o", r"$\hat{T}$", data, group="Obs. Type")


class Type(ParameterDiscrete):
    def __init__(self):
        super().__init__("type", "$T$", group="Type")

    def get_discrete(self, data):
        return "Ia", "II"


class PeakLuminosity(ParameterLatent):
    def __init__(self, n, estimated_zeros, estimated_redshift, lum):
        super().__init__("L", "$L0$", n, group="Peak Luminosity")
        self.estz = estimated_zeros
        self.estr = estimated_redshift
        self.lum = lum

    def get_suggestion_requirements(self):
        return ["c_o", "z_o"]

    def get_suggestion(self, data):
        return self.lum

    def get_suggestion_sigma(self, data):
        return 0.002 * self.lum


class PeakTime(ParameterLatent):
    def __init__(self, n, ts):
        super().__init__("t", r"$t0$", n, group="Peak Time")
        self.ts = ts

    def get_suggestion_requirements(self):
        return ["c_o", "t_o"]

    def get_suggestion(self, data):
        return self.ts

    def get_suggestion_sigma(self, data):
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


class LuminosityMeanIa(ParameterUnderlying):
    def __init__(self):
        super().__init__("mu_L1", r"$\mu_{LIa}$", group="SNIa")

    def get_log_prior(self, data):
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 500

    def get_suggestion_sigma(self, data):
        return 10


class LuminositySigmaIa(ParameterUnderlying):
    def __init__(self):
        super().__init__("sigma_L1", r"$\sigma_{LIa}$", group="SNIa")

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 50

    def get_suggestion_sigma(self, data):
        return 2

    def get_log_prior(self, data):
        if data["sigma_L1"] < 0:
            return -np.inf
        return 1


class StretchMeanIa(ParameterUnderlying):
    def __init__(self):
        super().__init__("mu_s1", r"$\mu_{sIa}$", group="SNIa")

    def get_log_prior(self, data):
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 10

    def get_suggestion_sigma(self, data):
        return 3


class StretchSigmaIa(ParameterUnderlying):
    def __init__(self):
        super().__init__("sigma_s1", r"$\sigma_{sIa}$", group="SNIa")

    def get_log_prior(self, data):
        if data["sigma_s1"] < 0:
            return -np.inf
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 2

    def get_suggestion_sigma(self, data):
        return 1


class LuminosityMeanII(ParameterUnderlying):
    def __init__(self):
        super().__init__("mu_L2", r"$\mu_{LII}$", group="SNII")

    def get_log_prior(self, data):
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 400

    def get_suggestion_sigma(self, data):
        return 10


class LuminositySigmaII(ParameterUnderlying):
    def __init__(self):
        super().__init__("sigma_L2", r"$\sigma_{LII}$", group="SNII")

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 20

    def get_suggestion_sigma(self, data):
        return 2

    def get_log_prior(self, data):
        if data["sigma_L2"] < 0:
            return -np.inf
        return 1


class StretchMeanII(ParameterUnderlying):
    def __init__(self):
        super().__init__("mu_s2", r"$\mu_{sII}$", group="SNII")

    def get_log_prior(self, data):
        return 1

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return 15

    def get_suggestion_sigma(self, data):
        return 3


class StretchSigmaII(ParameterUnderlying):
    def __init__(self):
        super().__init__("sigma_s2", r"$\sigma_{sII}$", group="SNII")

    def get_log_prior(self, data):
        if data["sigma_s2"] < 0:
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


class Rate(ParameterUnderlying):
    def __init__(self):
        super().__init__("rate", "$r$", group="Rate")

    def get_suggestion_sigma(self, data):
        return 0.3

    def get_suggestion(self, data):
        return 0.6

    def get_log_prior(self, data):
        if data["rate"] < 0 or data["rate"] > 1:
            return -np.inf
        return 1


class ToRate(Edge):
    def __init__(self):
        super().__init__("type", "rate")

    def get_log_likelihood(self, data):
        ias = data["type"] == "Ia"
        r = data["rate"]
        return np.log(r * ias + (1 - r) * (1 - ias))


class ToType(Edge):
    def __init__(self):
        super().__init__("type_o", "type")

    def get_log_likelihood(self, data):
        same = data["type_o"] == data["type"]
        return np.log(0.3 + 0.4 * same)


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
        flux = lum / (z * z)
        return {"f": flux}


class ToLuminosity(EdgeTransformation):
    def __init__(self):
        super().__init__("lum", ["type", "L", "t", "t_o", "s"])

    def get_transformation(self, data):
        l0 = data["L"][:, None]
        s = data["s"][:, None]
        diff = data["t"][:, None] - data["t_o"]
        vals1 = np.exp(-(diff * diff) / (2 * s * s))
        vals2 = np.exp(-np.abs(diff) / (2 * s * s))
        mask = 1.0 * (data["type"] == "Ia")
        vals = mask[:, None] * vals1 + (1 - mask)[:, None] * vals2
        lums = l0 * vals
        return {"lum": lums}


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
        super().__init__("L", ["type", "mu_L1", "sigma_L1", "mu_L2", "sigma_L2"])
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        type_mask = data["type"] == "Ia"
        means = type_mask * data["mu_L1"] + (1 - type_mask) * data["mu_L2"]
        sigma = type_mask * data["sigma_L1"] + (1 - type_mask) * data["sigma_L2"]
        diff = data["L"] - means
        return -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)


class ToStretchDistribution(Edge):
    def __init__(self):
        super().__init__("s", ["type", "mu_s1", "sigma_s1", "mu_s2", "sigma_s2"])
        self.factor = np.log(np.sqrt(2 * np.pi))

    def get_log_likelihood(self, data):
        type_mask = data["type"] == "Ia"
        means = type_mask * data["mu_s1"] + (1 - type_mask) * data["mu_s2"]
        sigma = type_mask * data["sigma_s1"] + (1 - type_mask) * data["sigma_s2"]
        diff = data["s"] - means
        return -0.5 * diff * diff / (sigma * sigma) - self.factor - np.log(sigma)


class EfficiencyModelUncorrected(Model):
    def __init__(self, observed_counts, observed_redshift, observed_times, observed_types, observed_calibration,
                 observed_zero_points, actual_lum, actual_stretch, actual_t0, seed=0, name="Uncorrected"):
        super().__init__(name)
        np.random.seed(seed)
        # Observed Parameters
        self.add(ObservedRedshift(observed_redshift))
        self.add(ObservedCounts(observed_counts))
        self.add(ObservedTimes(observed_times))
        self.add(ObservedType(observed_types))

        # Latent Parameters
        self.add(PeakLuminosity(observed_redshift.size, observed_zero_points, observed_redshift, actual_lum))
        self.add(PeakTime(observed_redshift.size, actual_t0))
        self.add(Stretch(observed_redshift.size, actual_stretch))

        # Transformed Parameters
        self.add(Type())
        self.add(Luminosity())
        self.add(Flux())
        self.add(Counts())

        # Underlying parameters
        self.add(LuminosityMeanIa())
        self.add(LuminositySigmaIa())
        self.add(LuminosityMeanII())
        self.add(LuminositySigmaII())
        self.add(StretchMeanIa())
        self.add(StretchSigmaIa())
        self.add(StretchMeanII())
        self.add(StretchSigmaII())
        self.add(Rate())
        self.add(Calibration(observed_zero_points, observed_calibration))

        # Transformed Edges
        self.add(ToLuminosity())
        self.add(ToCounts())
        self.add(ToFlux())

        # Edges
        self.add(ToRate())
        self.add(ToType())
        self.add(ToLuminosityDistribution())
        self.add(ToStretchDistribution())
        self.add(ToObservedCounts())

        self.finalise()


def get_weights(mul, sigmal, mul2, sigmal2, mus, sigmas, mus2, sigmas2, rate, z0, z1, threshold, n=1e4):
    n = int(n)
    type = np.random.random(size=n) < rate
    ls1 = np.random.normal(loc=mul, scale=sigmal, size=n)
    ls2 = np.random.normal(loc=mul2, scale=sigmal2, size=n)
    ss1 = np.random.normal(loc=mus, scale=sigmas, size=n)
    ss2 = np.random.normal(loc=mus2, scale=sigmas2, size=n)
    ls = type * ls1 + (1 - type) * ls2
    ss = type * ss1 + (1 - type) * ss2
    zs = np.random.uniform(0.6, high=1.5, size=n)

    ts = np.arange(-30, 30, 1)
    top1 = -(ts * ts)
    top2 = -(ts * ts)**0.5
    interior1 = top1[:, None] / (2 * ss * ss)[None, :]
    interior2 = top2[:, None] / (2 * ss * ss)[None, :]
    fluxes1 = (ls / (zs * zs))[None, :] * np.exp(interior1)
    fluxes2 = (ls / (zs * zs))[None, :] * np.exp(interior2)

    fluxes = type * fluxes1 + (1 - type) * fluxes2

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


def get_data(seed=5, n=50):
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
    rate = 0.6
    screwup = 0.3
    lmu = 500
    lmu2 = 300
    lsigma2 = 30
    lsigma = 50
    smu = 10
    smu2 = 15
    ssigma = 2
    ssigma2 = 2

    # Data
    zs = []
    ts = []
    ls = []
    cs = []
    mask = []
    ss = []
    t0s = []
    types = []
    for i in range(n):
        t0 = np.random.randint(low=1, high=300)
        t = np.arange(-t_sep * (num_obs // 2), t_sep * (num_obs // 2), t_sep)
        type = np.random.random() < rate
        if type:
            types.append("Ia")
            l0 = np.random.normal(loc=lmu, scale=lsigma)
            s = np.random.normal(loc=smu, scale=ssigma)
            z = np.random.uniform(0.6, 1.5)
            lums = l0 * np.exp(-(t * t) / (2 * s * s))
        else:
            types.append("II")
            l0 = np.random.normal(loc=lmu2, scale=lsigma2)
            s = np.random.normal(loc=smu2, scale=ssigma2)
            z = np.random.uniform(0.1, 1.5)
            lums = l0 * np.exp(-(t * t)**0.5 / (2 * s * s))
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
    for i, type in enumerate(types):
        if np.random.random() < screwup:
            if type == "Ia":
                types[i] = "II"
            elif type == "II":
                types[i] = "Ia"

    zs = np.array(zs)
    ts = np.array(ts)
    cs = np.array(cs)
    ss = np.array(ss)
    ls = np.array(ls)
    t0s = np.array(t0s)
    mask = np.array(mask)
    print(mask.sum(), n, mask.sum()/n, ls.mean(), ls[mask].mean(), np.std(ls), np.std(ls[mask]))

    return lmu, lsigma, lmu2, lsigma2, smu, ssigma, smu2, ssigma2, rate, zeros, calibration, threshold, ls, ss, t0s, zs, ts, cs, mask, num_obs, types


def plot_data(dir_name):
    lmu, lmu2, lsigma, lsigma2, smu, smu2, ssigma, ssigma2, rate, zeros, calibration, threshold, ls, ss, t0s, zs, ts, cs, mask, num_obs, types = get_data(n=2000)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel("$z$")
    ax.set_ylabel("$L$")
    print(types)
    ax.scatter(zs[mask], ls[mask], color="#1976D2", alpha=0.7, label="Discovered")
    ax.scatter(zs[~mask], ls[~mask], color="#D32F2F", alpha=0.7, label="Undiscovered")
    ax.axhline(lmu, color="k", ls="--")
    ax.legend(loc=3, fontsize=12)
    fig.savefig(os.path.abspath(dir_name + "/output/data.png"), bbox_inces="tight", dpi=300)

def get_weight_from_row(row, threshold):
    return get_weights(row[0], row[1], row[2], row[3], row[4], row[5], row[6],
                row[7], row[8], row[9], row[10], threshold)

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    dir_name = os.path.dirname(__file__)
    t = os.path.abspath(dir_name + "/output/data_%s")
    plot_file = os.path.abspath(dir_name + "/output/surfaces.png")
    walk_file = os.path.abspath(dir_name + "/output/walk_%s.png")

    # plot_data(dir_name)
    # lmu, lmu2, lsigma, lsigma2, smu, smu2, ssigma, ssigma2, rate, zeros, calibration, threshold, ls, ss, t0s, zs, ts, cs, mask, num_obs, types = get_data(n=1000)
    # model = EfficiencyModelUncorrected(cs, zs, ts, types, calibration, zeros, ls, ss, t0s, )
    # pgm_file = os.path.abspath(dir_name + "/output/pgm.png")
    # fig = model.get_pgm(pgm_file, seed=5)
    c = ChainConsumer()
    n = 1
    w = 4
    colours = ["#4CAF50", "#D32F2F", "#1E88E5"] * n
    for i in range(n):
        lmu, lmu2, lsigma, lsigma2, smu, smu2, ssigma, ssigma2, rate, zeros, calibration, threshold, ls, ss, t0s, zs, ts, cs, mask, num_obs, types = get_data(seed=i)
        theta = [lmu, lmu2, lsigma, lsigma2, smu, smu2, ssigma, ssigma2, rate] + zeros.tolist() + ls.tolist() + t0s.tolist() + ss.tolist()
        theta2 = theta + ls.tolist() + t0s.tolist() + ss.tolist()
        kwargs = {"num_steps": 5000, "num_burn": 250000, "save_interval": 60, "plot_covariance": True, "covariance_adjust": 10000}
        sampler = BatchMetropolisHastings(num_walkers=w, kwargs=kwargs, temp_dir=t % i, num_cores=4)

        model_good = EfficiencyModelUncorrected(cs, zs, ts, types, calibration, zeros, ls, ss, t0s, name="Good%d" % i)

        model_good.fit(sampler, chain_consumer=c)
        mtypes = [t for t, m in zip(types, mask) if m]
        model_un = EfficiencyModelUncorrected(cs[mask], zs[mask], ts[mask], mtypes, calibration,
                                              zeros, ls[mask], ss[mask], t0s[mask], name="Uncorrected%d" % i)
        model_un.fit(sampler, chain_consumer=c)

        biased_chain = c.chains[-1]
        filename = dir_name + "/output/weights.txt"
        if not os.path.exists(filename):
            weights = Parallel(n_jobs=4, verbose=100, batch_size=100)(delayed(get_weight_from_row)(row, threshold) for row in biased_chain)
            weights = np.array(weights)
            np.savetxt(filename, weights)
        else:
            weights = np.loadtxt(filename)
        weights = (1 / np.power(weights, mask.sum()))
        c.add_chain(biased_chain, name="Importance Sampled", weights=weights)

    c.configure_bar(shade=True)
    c.configure_general(bins=1.0, colours=colours)
    c.configure_contour(sigmas=[0, 0.01, 1, 2], contourf=True, contourf_alpha=0.2)
    c.plot(filename=plot_file, truth=theta, figsize=(10, 10), legend=False, parameters=10)
    for i in range(len(c.chains)):
        c.plot_walks(filename=walk_file % c.names[i], chain=i, truth=theta)
        # c.divide_chain(i, w).configure_general(rainbow=True) \
        #     .plot(figsize=(5, 5), filename=plot_file.replace(".png", "_%s.png" % c.names[i]),
        #           truth=theta)
