import numpy as np
import os
import inspect
import pickle

from scipy.interpolate import interp1d
from scipy.stats import binned_statistic

from dessn.framework.simulation import Simulation
from dessn.general.pecvelcor import get_sigma_mu_pecvel
from dessn.framework.simulations.selection_effects import des_sel, lowz_sel


class SNANASimulation(Simulation):
    def __init__(self, num_supernova, sim_name, num_nodes=4, use_sim=False, cov_scale=1.0, global_calib=13, shift=None, type="G10", kappa=0.0, bias_cor=True, zlim=None):
        super().__init__()
        self.simulation_name = sim_name
        self.type = type
        self.global_calib = global_calib
        this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        self.data_folder = this_dir + "/snana_data/%s/" % self.simulation_name
        assert os.path.exists(self.data_folder), "Cannot find folder %s" % self.data_folder
        self.use_sim = use_sim
        self.num_nodes = num_nodes
        self.systematic_labels = None
        self.kappa = kappa
        self.num_supernova = num_supernova
        self.cov_scale = cov_scale
        self.manual_selection = None
        self.shift = shift
        self.get_systematic_names()
        self.num_calib = len(self.systematic_labels)
        self.bias_cor = bias_cor
        self.zlim = zlim
        if self.num_calib == 0:
            self.num_calib = 1

    def get_correction(self, cov_scale=1.0):
        if "_des" in self.simulation_name.lower():
            self.logger.info("Getting DES correction for sim %s" % self.simulation_name)
            return des_sel(cov_scale=cov_scale, shift=self.shift, type=self.type, kappa=self.kappa, zlim=self.zlim)
        elif "_lowz" in self.simulation_name.lower():
            self.logger.info("Getting LOWZ correction for sim %s" % self.simulation_name)
            return lowz_sel(cov_scale=cov_scale, shift=self.shift, type=self.type, kappa=self.kappa, zlim=self.zlim)
        else:
            raise ValueError("Cannot find des or lowz in your sim name, unsure which selection function to use!")

    def get_name(self):
        return self.simulation_name

    def get_truth_values(self):
        return [
            ("Om", 0.3, r"$\Omega_m$"),
            ("Ol", 0.7, r"$\Omega_\Lambda$"),
            ("w", -1.0, r"$w$", True, -1.5, -0.5),
            ("alpha", 0.14, r"$\alpha$"),
            ("alpha_z", 0.0, r"$\alpha_z$"),
            ("beta", 3.1, r"$\beta$"),
            ("beta_z", 0.0, r"$\beta_z$"),
            ("mean_MB", -19.365, r"$\langle M_B \rangle$"),
            ("mean_x1", np.zeros(self.num_nodes), r"$\langle x_1^{%d} \rangle$"),
            ("mean_c", np.zeros(self.num_nodes), r"$\langle c^{%d} \rangle$"),
            ("sigma_MB", 0.1, r"$\sigma_{\rm m_B}$"),
            ("sigma_x1", 1.0, r"$\sigma_{x_1}$"),
            ("sigma_c", 0.1, r"$\sigma_c$"),
            ("log_sigma_MB", np.log(0.1), r"$\log\sigma_{\rm m_B}$"),
            ("log_sigma_x1", np.log(0.5), r"$\log\sigma_{x_1}$"),
            ("log_sigma_c", np.log(0.1), r"$\log\sigma_c$"),
            ("alpha_c", 0, r"$\alpha_c$"),
            ("dscale", 0, r"$\delta(0)$"),
            ("dratio", 0.5, r"$\delta(\infty)/\delta(0)$"),
            ("intrinsic_correlation", np.identity(3), r"$\rho$"),
            ("calibration", np.zeros(self.num_calib), r"$\delta \mathcal{Z}_%d$")
        ]

    def get_systematic_names(self):
        if self.systematic_labels is None:
            this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
            filename = this_dir + "/snana_data/%s/sys_names.pkl" % self.simulation_name
            with open(filename, 'rb') as f:
                names = pickle.load(f)
            self.systematic_labels = names
        return self.systematic_labels

    def get_all_supernova(self, n_sne, cosmology_index=0):
        self.logger.info("Getting SNANA data from %s" % self.data_folder)

        supernovae_files = [np.load(self.data_folder + "/" + f) for f in os.listdir(self.data_folder) if f.startswith("all")]
        supernovae = np.vstack(tuple(supernovae_files))
        passed = supernovae > 100
        mags = supernovae - 100 * passed.astype(np.int)
        res = {
            "sim_apparents": mags,
            "passed": passed
        }
        return res

    def get_bias_cor(self, des=True):
        print("Getting biascor for des=%s" % des)
        if des:
            file = self.data_folder + "../DES3YR_DES_BHMEFF_AM%s/passed_0.npy"
        else:
            file = self.data_folder + "../DES3YR_LOWZ_BHMEFF_%s/passed_0.npy"
        models = ["C11", "G10"]
        bine = 30
        means = []
        for model in models:
            data = np.load(file % model)
            z = data[:, 1]
            c_obs = data[:, 8]
            c_true = data[:, 5]
            diff = c_obs - c_true
            mean, bine, _ = binned_statistic(z, diff, bins=bine)
            means.append(mean)
            std, _, _ = binned_statistic(z, diff, statistic=lambda x: np.std(x) / np.sqrt(len(x)), bins=bine)

        middle = np.array(means)
        binc = 0.5 * (bine[1:] + bine[:-1])
        return binc, models[0], middle[0]

    def get_passed_supernova(self, n_sne, cosmology_index=0):
        filename = self.data_folder + "passed_%d.npy" % cosmology_index
        assert os.path.exists(filename), "Cannot find file %s, do you have this realisations?" % filename
        supernovae = np.load(filename)
        self.logger.info("%s SN in %s" % (supernovae.shape[0], filename))

        if self.zlim is not None:
            redshifts = supernovae[:, 1]
            self.logger.info("Enforcing zlim of %0.2f" % self.zlim)
            mask = redshifts < self.zlim
            self.logger.info("%d supernova out of %d passed the redshift cut" % (mask.sum(), supernovae.shape[0]))
            supernovae = supernovae[mask, :]

        if n_sne != -1:
            supernovae = supernovae[:n_sne, :]
        else:
            n_sne = supernovae.shape[0]
            self.logger.info("All SN requested: found %d SN" % n_sne)
        cids = supernovae[:, 0]
        redshifts = supernovae[:, 1]
        masses = supernovae[:, 2]
        s_ap = supernovae[:, 3]
        s_st = supernovae[:, 4]
        s_co = supernovae[:, 5]
        apparents = supernovae[:, 6]
        stretches = supernovae[:, 7]
        colours = supernovae[:, 8]
        bias_mB = supernovae[:, 9]
        bias_x1 = supernovae[:, 10]
        bias_c = supernovae[:, 11]
        extra_uncert = get_sigma_mu_pecvel(redshifts)
        obs_mBx1c_cov, obs_mBx1c, deta_dcalibs = [], [], []

        shift_amount = np.zeros(redshifts.shape)
        shift_deltas = np.zeros(redshifts.shape)
        if self.bias_cor:
            apparents -= bias_mB
            stretches -= bias_x1
            colours -= bias_c
            # cor_z, cor_models, cor_means = self.get_bias_cor("_DES" in self.simulation_name)
            # shift0 = cor_means[0]
            # shift_amount = interp1d(cor_z, shift0, bounds_error=False, fill_value=(shift0[0], shift0[-1]))(redshifts)
            # delta = cor_means[1] - cor_means[0]
            # shift_deltas = interp1d(cor_z, delta, bounds_error=False, fill_value=(delta[0], delta[-1]))(redshifts)
            # shift_deltas = interp1d(cor_z, cor_means, bounds_error=False, fill_value=(cor_means[0], cor_means[-1]))(redshifts)

        for i, (mb, x1, c, smb, sx1, sc, eu, sa) in enumerate(zip(apparents, stretches, colours, s_ap, s_st, s_co, extra_uncert, shift_amount)):
            if self.use_sim:
                cov = np.diag(np.array([0.04, 0.1, 0.04]) ** 2)
                vector = np.array([smb, sx1, sc]) + np.random.multivariate_normal([0, 0, 0], cov)
            else:
                vector = np.array([mb, x1, c - sa])
                cov = supernovae[i, 12:12 + 9].reshape((3, 3))
            cov[0, 0] += eu**2
            calib = supernovae[i, 12+9:].reshape((3, -1))
            obs_mBx1c_cov.append(cov)
            obs_mBx1c.append(vector)
            deta_dcalibs.append(calib)
        covs = np.array(obs_mBx1c_cov)
        deta_dcalibs = np.array(deta_dcalibs)
        obs_mBx1c = np.array(obs_mBx1c)
        result = {
            "cids": cids,
            "n_sne": n_sne,
            "obs_mBx1c": obs_mBx1c,
            "obs_mBx1c_cov": covs,
            "deta_dcalib": deta_dcalibs,
            "redshifts": redshifts,
            "masses": masses,
            "sim_apparents": s_ap,
            "sim_stretches": s_st,
            "sim_colours": s_co,
            "shift_deltas": shift_deltas,
            "prob_ia": np.ones(n_sne) * 0.999999
        }
        return result

    def get_approximate_correction(self):
        if self.manual_selection is None:
            self.manual_selection = self.get_correction(cov_scale=self.cov_scale)
        return self.manual_selection
