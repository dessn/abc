import numpy as np
from astropy.cosmology import FlatwCDM
from scipy.stats import norm, multivariate_normal, skewnorm

from dessn.framework.simulation import Simulation


class SimpleSimulation(Simulation):

    def __init__(self, num_supernova, dscale=0.08, alpha_c=5, mass=True, num_nodes=4, lowz=False, min_prob_ia=1.0):
        super().__init__()
        self.alpha_c = alpha_c
        self.dscale = dscale
        self.min_prob_ia = min_prob_ia
        self.num_calib = 1
        if lowz:
            self.skewnorm = True
            self.mb_alpha = 5.87
            self.mb_mean = 13.72
            self.mb_width = 1.35
            self.power = 0.75
            self.max_z_gen = 0.2
        else:
            self.skewnorm = False
            self.mb_alpha = 4
            self.mb_mean = 22.14
            self.mb_width = 0.65
            self.power = 0.5
            self.max_z_gen = 0.8

        self.mass_scale = 1.0 if mass else 0.0
        self.num_nodes = num_nodes
        self.num_supernova = num_supernova

    def get_name(self):
        return "simple"

    def get_truth_values(self):
        return [
            ("Om", 0.3, r"$\Omega_m$"),
            ("Ol", 0.7, r"$\Omega_\Lambda$"),
            ("w", -1.0, r"$w$", True, -1.5, -0.5),
            ("alpha", 0.14, r"$\alpha$"),
            ("delta_alpha", 0, r"$\delta_\alpha$"),
            ("beta", 3.1, r"$\beta$"),
            ("delta_beta", 0, r"$\delta_\beta$"),
            ("mean_MB", -19.365, r"$\langle M_B \rangle$"),
            ("outlier_MB_delta", 2, r"$\delta M_B$"),
            ("outlier_dispersion", np.array([1.0, 1.0, 0.7]), r"$\sigma_{\rm out}^{%d}$"),
            ("mean_x1", np.zeros(self.num_nodes), r"$\langle x_1^{%d} \rangle$"),
            ("mean_c", np.zeros(self.num_nodes), r"$\langle c^{%d} \rangle$"),
            ("sigma_MB", 0.1, r"$\sigma_{\rm m_B}$"),
            ("sigma_x1", 1.0, r"$\sigma_{x_1}$"),
            ("sigma_c", 0.1, r"$\sigma_c$"),
            ("log_sigma_MB", np.log(0.1), r"$\log\sigma_{\rm m_B}$"),
            ("log_sigma_x1", np.log(0.5), r"$\log\sigma_{x_1}$"),
            ("log_sigma_c", np.log(0.1), r"$\log\sigma_c$"),
            ("alpha_c", self.alpha_c, r"$\alpha_c$"),
            ("dscale", self.dscale, r"$\delta(0)$"),
            ("dratio", 0.5, r"$\delta(\infty)/\delta(0)$"),
            ("intrinsic_correlation", np.identity(3), r"$\rho$"),
            ("calibration", np.zeros(self.num_calib), r"$\delta \mathcal{Z}_%d$")
        ]

    def get_all_supernova(self, n_sne, cosmology_index=0):
        truth = self.get_truth_values_dict()
        self.logger.info("Generating simple data for %d supernova, with skewness %d..." % (n_sne, truth["alpha_c"]))
        np.random.seed(cosmology_index)
        self.logger.info("Generating for cosmology index %d" % cosmology_index)
        cosmology = FlatwCDM(70.0, truth["Om"])

        # Unwrap some values
        alpha, beta, dscale, dratio = truth["alpha"], truth["beta"], truth["dscale"], truth["dratio"]
        sim_mBx1c, obs_mBx1c_cov, obs_mBx1c, deta_dcalib = [], [], [], []
        redshifts_all, redshift_pre_comp_all, p_high_masses_all, mask_all, mbs_all,  = [], [], [], [], []
        ia_probs_all = []
        sim_x1s_all, sim_cs_all = [], []

        # Assume constant population.
        means = np.array([truth["mean_MB"], truth["mean_x1"][0], truth["mean_c"][0]])
        sigmas = np.array([truth["sigma_MB"], truth["sigma_x1"], truth["sigma_c"]])
        sigmas_mat = np.dot(sigmas[:, None], sigmas[None, :])
        correlations = np.dot(truth["intrinsic_correlation"], truth["intrinsic_correlation"].T)
        pop_cov = correlations * sigmas_mat
        probs = []

        outlier_MB_delta, outlier_dispersion = truth["outlier_MB_delta"], truth["outlier_dispersion"]
        nn = 2000
        # Generate 1000 at a time
        while True:
            redshifts = (np.random.uniform(0.02, self.max_z_gen, nn) ** self.power)
            dist_mod = cosmology.distmod(redshifts).value
            redshift_pre_comp = 0.9 + np.power(10, 0.95 * redshifts)
            p_high_masses = np.random.uniform(low=0.0, high=1.0, size=dist_mod.size) * self.mass_scale
            ia_probs = np.random.uniform(low=self.min_prob_ia, high=1.0, size=nn)
            contaminations = np.random.random(nn) > ia_probs
            for zz, mu, p, contamination in zip(redshift_pre_comp, dist_mod, p_high_masses, contaminations):
                while True:
                    MB, x1, c = np.random.multivariate_normal(means, pop_cov)
                    if np.random.random() < norm.cdf(truth["alpha_c"] * (c - truth["mean_c"][0]) / truth["sigma_c"], 0, 1):
                        skew_prob = norm.logcdf(truth["alpha_c"] * (c - truth["mean_c"][0]) / truth["sigma_c"], 0, 1)
                        break
                probs.append(multivariate_normal.logpdf([MB, x1, c], mean=means, cov=pop_cov) + skew_prob)
                if contamination:
                    MB -= outlier_MB_delta
                    adjust = np.random.normal(loc=0, scale=np.sqrt(outlier_dispersion**2 - sigmas**2), size=3)
                    MB += adjust[0]
                    x1 += adjust[1]
                    c += adjust[2]

                mass_correction = dscale * (1.9 * (1 - dratio) / zz + dratio)
                mb = MB + mu - alpha * x1 + beta * c - mass_correction * p
                vector = np.array([mb, x1, c])
                # Add intrinsic scatter to the mix
                diag = np.array([0.04, 0.2, 0.03]) ** 2
                cov = np.diag(diag)
                sim_mBx1c.append(vector)
                vector += np.random.multivariate_normal([0, 0, 0], cov)
                obs_mBx1c_cov.append(cov)
                obs_mBx1c.append(vector)
                deta_dcalib.append(np.random.normal(0, 3e-3, size=(3, self.num_calib)))
            redshifts_all += redshifts.tolist()
            redshift_pre_comp_all += redshift_pre_comp.tolist()
            p_high_masses_all += p_high_masses.tolist()
            ia_probs_all += ia_probs.tolist()

            mbs = np.array([o[0] for o in sim_mBx1c[-nn:]])
            sim_x1 = np.array([o[1] for o in sim_mBx1c[-nn:]])
            sim_c = np.array([o[2] for o in sim_mBx1c[-nn:]])
            vals = np.random.uniform(size=mbs.size)
            if not self.skewnorm:
                pdfs = 1 - norm.cdf(mbs, self.mb_mean, self.mb_width)
                pdfs /= pdfs.max()
            else:
                pdfs = skewnorm.pdf(mbs, self.mb_alpha, self.mb_mean, self.mb_width)
                pdfs /= pdfs.max()
            mask = vals < pdfs
            mbs_all += mbs.tolist()
            sim_x1s_all += sim_x1.tolist()
            sim_cs_all += sim_c.tolist()
            mask_all += mask.tolist()

            self.logger.debug("Have %d passed out of required %d sne, generated %d"
                              % (np.array(mask_all).sum(), n_sne, len(mask_all)))
            if np.array(mask_all).sum() >= n_sne:
                break

        indexes = np.array(mask_all).cumsum()
        cut_index = np.where(indexes == n_sne)[0][0] + 1
        mask_all = np.array(mask_all)
        self.logger.debug("Generated %d objects out of %d passed, %d percent" % (mask_all.sum(), mask_all.size, 100 * (mask_all.sum() / mask_all.size)))

        return {
            "n_sne": n_sne,
            "obs_mBx1c": np.array(obs_mBx1c[:cut_index]),
            "obs_mBx1c_cov": np.array(obs_mBx1c_cov[:cut_index]),
            "deta_dcalib": np.array(deta_dcalib[:cut_index]),
            "redshifts": np.array(redshifts_all[:cut_index]),
            "masses": np.array(p_high_masses_all[:cut_index]),
            "existing_prob": np.array(probs[:cut_index]),
            "sim_apparents": np.array(mbs_all[:cut_index]),
            "sim_stretches": np.array(sim_x1s_all[:cut_index]),
            "sim_colours": np.array(sim_cs_all[:cut_index]),
            "passed": np.array(mask_all[:cut_index]),
            "prob_ia": np.array(ia_probs_all[:cut_index])
        }

    def get_approximate_correction(self):
        if not self.skewnorm:
            print("ccdf approx")
            return self.mb_mean, self.mb_width, None, None
        else:
            print("skewnorm approx")
            return self.mb_mean, self.mb_width, self.mb_alpha, 1.0
