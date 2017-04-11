from scipy.interpolate import interp1d
import numpy as np

from dessn.framework.models.approx_model import ApproximateModel


class FullModel(ApproximateModel):

    def __init__(self, num_supernova, file="full.stan"):
        super().__init__(num_supernova, file=file)

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
            "sim_log_weight": np.log(probs * 2 * weights * np.diff(zs_sample)[0] / 3)
        }


    def get_data(self, simulation, cosmology_index, add_zs=None):
        return super().get_data(simulation, cosmology_index, add_zs=self.get_extra_zs)
