import numpy as np
import os
import inspect
from dessn.framework.simulation import Simulation


class SNANASimulation(Simulation):
    def __init__(self, num_supernova, real_data_name, simulation_name=None, num_nodes=4, use_sim=False, num_calib=4):
        super().__init__()
        self.real_data_name = real_data_name
        self.simulation_name = simulation_name
        self.use_sim = use_sim
        if self.simulation_name is None:
            self.simulation_name = self.real_data_name
        self.num_nodes = num_nodes
        self.num_supernova = num_supernova
        self.num_calib = num_calib

    def get_name(self):
        return "snana_%s" % self.real_data_name

    def get_truth_values(self):
        return [
            ("Om", 0.3, r"$\Omega_m$"),
            # ("w", -1.0, r"$w$", True, -1.5, -0.5),
            ("alpha", 0.14, r"$\alpha$"),
            ("beta", 3.1, r"$\beta$"),
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

    def get_passed_from_name(self, name, n_sne, cosmology_index=0):
        this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        data_folder = this_dir + "/snana_data/%s" % name
        self.logger.info("Getting SNANA data from %s" % name)
        if self.use_sim:
            self.logger.warn("Simulation using simulated values, not 'observed'")

        supernovae_files = [np.load(data_folder + "/" + f) for f in os.listdir(data_folder) if "passed" in f]
        supernovae = np.vstack(tuple(supernovae_files))

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
            "sim_colours": s_co
        }
        return result

    def get_all_supernova(self, n_sne, cosmology_index=0):
        name = self.simulation_name
        this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        data_folder = this_dir + "/snana_data/%s" % name
        self.logger.info("Getting SNANA data from %s" % name)

        supernovae_files = [np.load(data_folder + "/" + f) for f in os.listdir(data_folder) if "all" in f]
        supernovae = np.vstack(tuple(supernovae_files))
        return {
            "redshifts": supernovae[:, 0],
            "sim_apparents": supernovae[:, 1],
            "passed": supernovae[:, 2].astype(bool)
        }

    def get_passed_supernova(self, n_sne, simulation=True, cosmology_index=0):
        name = self.simulation_name if simulation else self.real_data_name
        return self.get_passed_from_name(name, n_sne=n_sne, cosmology_index=cosmology_index)


class SNANASimulationGauss0p3(SNANASimulation):
    def __init__(self, num_supernova, num_nodes=4, use_sim=False):
        super().__init__(num_supernova, "gauss0p3", num_nodes=num_nodes, use_sim=use_sim, num_calib=22)


class SNANASimulationGauss0p2(SNANASimulation):
    def __init__(self, num_supernova, num_nodes=4, use_sim=False):
        super().__init__(num_supernova, "gauss0p2", simulation_name="gauss0p3", num_nodes=num_nodes, use_sim=use_sim, num_calib=22)

    def get_truth_values(self):
        t = super().get_truth_values()
        t[[r[0] for r in t].index("Om")] = ("Om", 0.2, r"$\Omega_m$")
        return t


class SNANASimulationGauss0p4(SNANASimulation):
    def __init__(self, num_supernova, num_nodes=4, use_sim=False):
        super().__init__(num_supernova, "gauss0p4", simulation_name="gauss0p3", num_nodes=num_nodes, use_sim=use_sim, num_calib=22)

    def get_truth_values(self):
        t = super().get_truth_values()
        t[[r[0] for r in t].index("Om")] = ("Om", 0.4, r"$\Omega_m$")
        return t


class SNANASimulationSkewed0p2(SNANASimulation):
    def __init__(self, num_supernova, num_nodes=4, use_sim=False):
        super().__init__(num_supernova, "skewed0p2", simulation_name="gauss0p3", num_nodes=num_nodes, use_sim=use_sim, num_calib=22)

    def get_truth_values(self):
        t = super().get_truth_values()
        t[[r[0] for r in t].index("Om")] = ("Om", 0.2, r"$\Omega_m$")
        return t


class SNANASimulationLowzGauss0p3(SNANASimulation):
    def __init__(self, num_supernova, num_nodes=4, use_sim=False, manual_selection=None):
        self.manual_selection = manual_selection
        super().__init__(num_supernova, "lowz_gauss0p3", num_nodes=num_nodes, use_sim=use_sim, num_calib=58)

    def get_approximate_correction(self, plot=False):
        if self.manual_selection is None:
            return super().get_approximate_correction(plot=plot)
        else:
            return self.manual_selection[0], self.manual_selection[1], None, 1.0


class SNANASimulationLowzGauss0p2(SNANASimulation):
    def __init__(self, num_supernova, num_nodes=4, use_sim=False):
        super().__init__(num_supernova, "lowz_gauss0p2", simulation_name="lowz_gauss0p3", num_nodes=num_nodes, use_sim=use_sim, num_calib=58)

    def get_truth_values(self):
        t = super().get_truth_values()
        t[[r[0] for r in t].index("Om")] = ("Om", 0.2, r"$\Omega_m$")
        return t


class SNANASimulationLowzGauss0p4(SNANASimulation):
    def __init__(self, num_supernova, num_nodes=4, use_sim=False):
        super().__init__(num_supernova, "lowz_gauss0p4", simulation_name="lowz_gauss0p3", num_nodes=num_nodes, use_sim=use_sim, num_calib=58)

    def get_truth_values(self):
        t = super().get_truth_values()
        t[[r[0] for r in t].index("Om")] = ("Om", 0.4, r"$\Omega_m$")
        return t


class SNANASimulationLowzSkewed0p2(SNANASimulation):
    def __init__(self, num_supernova, num_nodes=4, use_sim=False):
        super().__init__(num_supernova, "lowz_skewed0p2", simulation_name="lowz_gauss0p3", num_nodes=num_nodes,
                         use_sim=use_sim, num_calib=58)

    def get_truth_values(self):
        t = super().get_truth_values()
        t[[r[0] for r in t].index("Om")] = ("Om", 0.2, r"$\Omega_m$")
        return t