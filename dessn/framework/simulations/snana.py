import numpy as np
import os
import inspect
import pickle
from dessn.framework.simulation import Simulation
from dessn.general.pecvelcor import get_sigma_mu_pecvel
from dessn.framework.simulations.selection_effects import des_sel, lowz_sel


class SNANASimulation(Simulation):
    def __init__(self, num_supernova, sim_name, num_nodes=4, use_sim=False, cov_scale=1.0, global_calib=13):
        super().__init__()
        self.simulation_name = sim_name
        self.global_calib = global_calib
        this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        self.data_folder = this_dir + "/snana_data/%s/" % self.simulation_name
        assert os.path.exists(self.data_folder), "Cannot find folder %s" % self.data_folder
        self.use_sim = use_sim
        self.num_nodes = num_nodes
        self.systematic_labels = self.get_systematic_names()
        self.num_calib = len(self.systematic_labels)
        self.num_supernova = num_supernova

        self.manual_selection = self.get_correction(cov_scale=cov_scale)

    def get_correction(self, cov_scale=1.0):
        if "des" in self.simulation_name.lower():
            return des_sel(cov_scale=cov_scale)
        elif "lowz" in self.simulation_name.lower():
            return lowz_sel(cov_scale=cov_scale)
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
        this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        filename = this_dir + "/snana_data/%s/sys_names.pkl" % self.simulation_name
        with open(filename, 'rb') as f:
            names = pickle.load(f)
        return names

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

    def get_passed_supernova(self, n_sne, cosmology_index=0):
        filename = self.data_folder + "passed_%d.npy" % cosmology_index
        assert os.path.exists(filename), "Cannot find file %s, do you have this realisations?" % filename
        supernovae = np.load(filename)

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
        extra_uncert = get_sigma_mu_pecvel(redshifts)
        print("Extra z uncert:\n\t%s\n\t%s" % (redshifts, extra_uncert))

        obs_mBx1c_cov, obs_mBx1c, deta_dcalibs = [], [], []
        for i, (mb, x1, c, smb, sx1, sc, eu) in enumerate(zip(apparents, stretches, colours, s_ap, s_st, s_co, extra_uncert)):
            if self.use_sim:
                cov = np.diag(np.array([0.02, 0.1, 0.02]) ** 2)
                vector = np.array([smb, sx1, sc]) + np.random.multivariate_normal([0, 0, 0], cov)
            else:
                vector = np.array([mb, x1, c])
                cov = supernovae[i, 9:9 + 9].reshape((3, 3))
            cov[0, 0] += eu**2
            calib = supernovae[i, 9 + 9:].reshape((3, -1))
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
            "prob_ia": np.ones(n_sne),
            # "passed": np.ones(n_sne, dtype=np.bool)
        }
        return result

    def get_approximate_correction(self):
        return self.manual_selection
