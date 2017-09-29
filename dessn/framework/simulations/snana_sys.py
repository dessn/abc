import inspect
import os
import pickle
import numpy as np
from dessn.framework.simulation import Simulation


class SNANASysSimulation(Simulation):
    def __init__(self, n_sne, sys_index=0, sim="des", manual_selection=None):
        super().__init__()
        self.sys_index = sys_index
        self.sim = sim
        self.manual_selection = manual_selection
        self.folder = "sys_data/%s/" % sim
        self.num_supernova = n_sne
    
    def get_truth_values(self):
        return [
            ("Om", 0.3, r"$\Omega_m$"),
            ("Ol", 0.7, r"$\Omega_\Lambda$"),
            ("w", -1.0, r"$w$"),
            ("alpha", 0.14, r"$\alpha$"),
            ("beta", 3.1, r"$\beta$"),
            ("mean_MB", -19.365, r"$\langle M_B \rangle$")
        ]
    
    def get_name(self):
        return "%s_%d" % (self.sim, self.sys_index)

    def get_systematic_names(self):
        this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        filename = this_dir + "/snana_data/%s/sys_names.pkl" % self.real_data_name
        with open(filename, 'rb') as f:
            names = pickle.load(f)
        return names

    def get_approximate_correction(self, plot=False, manual=None):
        if self.manual_selection is not None:
            return self.manual_selection
        else:
            return super().get_approximate_correction(plot=plot, manual=manual)

    def get_all_supernova(self, n_sne, cosmology_index=0):
        file = os.path.abspath(inspect.stack()[0][1])
        dir_name = os.path.dirname(file)
        filename = dir_name + os.sep + self.folder + "%d_%d.npy" % (self.sys_index, cosmology_index + 1)
        data = np.load(filename)[:n_sne, :]

        return {
            "n_sne": n_sne,
            "obs_mBx1c": data[:, 6:9],
            "obs_mBx1c_cov": np.array([row[9:].reshape((3, 3)) for row in data]),
            "deta_dcalib": np.zeros((n_sne, 3, 1)),
            "redshifts": data[:, 1],
            "masses": np.zeros(n_sne),
            "existing_prob": data[:, 2],
            "sim_apparents": data[:, 3],
            "sim_stretches": data[:, 4],
            "sim_colours": data[:, 5],
            "passed": np.ones(n_sne, dtype=np.bool),
            "prob_ia": np.ones(n_sne)
        }