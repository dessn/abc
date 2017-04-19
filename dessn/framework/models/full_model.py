from scipy.interpolate import interp1d
import numpy as np
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp

from dessn.framework.models.approx_model import ApproximateModel
from dessn.utility.get_cosmologies import get_cosmology_dictionary


class FullModel(ApproximateModel):

    def __init__(self, num_supernova, filename="full.stan", num_nodes=4):
        super().__init__(num_supernova, filename=filename, num_nodes=num_nodes)

    def get_extra_zs(self, simulation, n=201, buffer=0.2):
        assert n % 2 == 1, "n needs to be odd"

        supernovae = simulation.get_all_supernova(20000)
        redshift_passed = supernovae["redshifts"][supernovae["passed"]]

        # Need to determine the max redshift to sample to
        zs = supernovae["redshifts"]

        hist, bins = np.histogram(zs, bins=50, density=True)
        binc = 0.5 * (bins[1:] + bins[:-1])

        max_zs = min(np.max(zs), np.max(redshift_passed) + buffer)
        min_zs = max(0.05, np.min(redshift_passed))

        zs_sample = np.linspace(min_zs, max_zs, n)

        probs = interp1d(binc, hist, assume_sorted=True, bounds_error=False, fill_value="extrapolate")(zs_sample)

        weights = np.ones(n)
        weight_2_indexes = 2 * np.arange(1, (n // 2))
        weight_4_indexes = 2 * np.arange(1, (n // 2) + 1) - 1
        weights[weight_2_indexes] += 1
        weights[weight_4_indexes] += 3
        return {
            "n_sim": n,
            "sim_redshifts": zs_sample,
            "sim_log_weight": np.log(probs * 2 * weights * np.diff(zs_sample)[0] / 3),
            "sim_redshift_pre_comp": 0.9 + np.power(10, 0.95 * zs_sample)
        }

    def get_data(self, simulation, cosmology_index, add_zs=None):
        return super().get_data(simulation, cosmology_index, add_zs=self.get_extra_zs)

    def correct_chain(self, chain_dictionary, simulation, data):
        self.logger.info("Starting full corrections")
        self.logger.info("Getting supernovae")
        supernovae = simulation.get_passed_supernova(n_sne=30000)

        self.logger.info("Getting cosmologies")
        cosmologies = get_cosmology_dictionary()

        # Unpack passed supernova details
        redshifts = supernovae["redshifts"]
        apparents = supernovae["sim_apparents"]
        stretches = supernovae["sim_stretches"]
        colours = supernovae["sim_colours"]
        existing_prob = supernovae["existing_prob"]
        masses = supernovae["masses"]

        nodes = data["nodes"]

        self.logger.info("Getting node weights")
        node_weights = self.get_node_weights(nodes, redshifts)

        using_log = "log_sigma_MB" in list(chain_dictionary.keys())

        weight = []
        # Iterate through our Stan chain
        self.logger.info("Iterating through chain")
        for i in range(chain_dictionary["mean_MB"].size):

            # Get the cosmology for Omega_m
            om = np.round(chain_dictionary["Om"][i], decimals=3)
            key = "%0.3f" % om
            mus = cosmologies[key](redshifts)

            # If mass is present in the output results, compute the mass correction
            if "dscale" in chain_dictionary.keys() and "dratio" in chain_dictionary.keys():
                dscale = chain_dictionary["dscale"][i]
                dratio = chain_dictionary["dratio"][i]
                redshift_pre_comp = 0.9 + np.power(10, 0.95 * redshifts)
                mass_correction = dscale * (1.9 * (1 - dratio) / redshift_pre_comp + dratio)
            else:
                mass_correction = 0
            mabs = apparents - mus + chain_dictionary["alpha"][i] * stretches \
                   - chain_dictionary["beta"][i] * colours + mass_correction * masses

            chain_MB = chain_dictionary["mean_MB"][i]
            chain_x1s = chain_dictionary["mean_x1"][i]
            chain_cs = chain_dictionary["mean_c"][i]

            chain_x1 = np.dot(node_weights, chain_x1s)
            chain_c = np.dot(node_weights, chain_cs)

            mbx1cs = np.vstack((mabs - chain_MB, stretches - chain_x1, colours - chain_c)).T

            if using_log:
                chain_sigmas = np.array(
                    [np.exp(chain_dictionary["log_sigma_MB"][i]),
                     np.exp(chain_dictionary["log_sigma_x1"][i]),
                     np.exp(chain_dictionary["log_sigma_c"][i])])
            else:
                chain_sigmas = np.array(
                    [chain_dictionary["sigma_MB"][i],
                     chain_dictionary["sigma_x1"][i],
                     chain_dictionary["sigma_c"][i]])

            chain_sigmas_mat = np.dot(chain_sigmas[:, None], chain_sigmas[None, :])
            chain_correlations = np.dot(chain_dictionary["intrinsic_correlation"][i],
                                        chain_dictionary["intrinsic_correlation"][i].T)
            chain_pop_cov = chain_correlations * chain_sigmas_mat
            chain_mean = np.array([0, 0, 0])

            chain_prob = multivariate_normal.logpdf(mbx1cs, chain_mean, chain_pop_cov)

            reweight = logsumexp(chain_prob - existing_prob)
            weight.append(reweight)

        self.logger.info("Finalising weights")
        weights = np.array(weight)
        existing = chain_dictionary.get("weight")
        if weights is None:
            weights = np.ones(weights.size)
        reweight = existing - data["n_sne"] * weights

        self.logger.info("Normalising weights")
        reweight -= reweight.mean()

        chain_dictionary["new_weight"] = reweight
        del chain_dictionary["intrinsic_correlation"]
        return chain_dictionary
