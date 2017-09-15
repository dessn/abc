import inspect
import os
import numpy as np
from dessn.framework.simulation import Simulation


class SNANABulkSimulation(Simulation):
    def __init__(self, n_sne, sim="des", manual_selection=None, use_sim=False, num_nodes=4, num_calib=0):
        super().__init__()
        self.sim = sim
        self.manual_selection = manual_selection
        self.folder = "bulk_data/%s/" % sim
        self.num_supernova = n_sne
        self.use_sim = use_sim
        self.num_nodes = num_nodes
        self.num_calib = num_calib
    
    def get_truth_values(self):
        return [
            ("Om", 0.3, r"$\Omega_m$"),
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
            ("alpha_x1", 0, r"$\alpha_c$"),
            ("dscale", 0, r"$\delta(0)$"),
            ("dratio", 0.5, r"$\delta(\infty)/\delta(0)$"),
            ("intrinsic_correlation", np.identity(3), r"$\rho$"),
            ("calibration", np.zeros(self.num_calib), r"$\delta \mathcal{Z}_%d$")
        ]
    
    def get_name(self):
        return self.sim

    def get_approximate_correction(self, plot=False, manual=None):
        if self.manual_selection is not None:
            return self.manual_selection
        else:
            raise ValueError("Bulk sims must specify manual selection")

    def get_all_supernova(self, n_sne, cosmology_index=0):
        file = os.path.abspath(inspect.stack()[0][1])
        dir_name = os.path.dirname(file)
        filename = dir_name + os.sep + self.folder + "passed.npy"
        supernovae = np.load(filename)
        np.random.seed(cosmology_index)
        if cosmology_index:
            self.logger.debug("Shuffling data for cosmology index %d" % cosmology_index)
            np.random.shuffle(supernovae)

        supernovae = supernovae[:n_sne, :]

        redshifts = supernovae[:, 1]
        apparents = supernovae[:, 6]
        stretches = supernovae[:, 7]
        colours = supernovae[:, 8]
        existing_prob = supernovae[:, 2]
        s_ap = supernovae[:, 3]
        s_st = supernovae[:, 4]
        s_co = supernovae[:, 5]
        masses = np.zeros(supernovae[:, 1].shape)
        obs_mBx1c_cov, obs_mBx1c, deta_dcalibs = [], [], []
        for i, (mb, x1, c, smb, sx1, sc) in enumerate(zip(apparents, stretches, colours, s_ap, s_st, s_co)):
            if self.use_sim:
                cov = np.diag(np.array([0.02, 0.1, 0.02]) ** 2)
                vector = np.array([smb, sx1, sc]) + np.random.multivariate_normal([0, 0, 0], cov)
            else:
                vector = np.array([mb, x1, c])
                cov = supernovae[i, 9:9 + 9].reshape((3, 3))
            calib = supernovae[i, 9 + 9:].reshape((3, -1))
            obs_mBx1c_cov.append(cov)
            obs_mBx1c.append(vector)
            deta_dcalibs.append(calib)
        covs = np.array(obs_mBx1c_cov)
        deta_dcalibs = np.array(deta_dcalibs)
        obs_mBx1c = np.array(obs_mBx1c)
        result = {
            "n_sne": n_sne,
            "obs_mBx1c": obs_mBx1c,
            "obs_mBx1c_cov": covs,
            "deta_dcalib": deta_dcalibs,
            "redshifts": redshifts,
            "masses": masses,
            "existing_prob": existing_prob,
            "sim_apparents": s_ap,
            "sim_stretches": s_st,
            "sim_colours": s_co,
            "prob_ia": np.ones(n_sne),
            "passed": np.ones(n_sne, dtype=np.bool)
        }
        return result