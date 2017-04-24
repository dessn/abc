from collections import OrderedDict

import numpy as np
import inspect
import os
from numpy.random import uniform, normal
from scipy.interpolate import interp1d

from dessn.framework.model import Model


class ApproximateModel(Model):

    def __init__(self, num_supernova, filename="approximate.stan", num_nodes=4):
        file = os.path.abspath(inspect.stack()[0][1])
        directory = os.path.dirname(file)
        stan_file = directory + "/stan/" + filename
        super().__init__(stan_file)

        self.num_redshift_nodes = num_nodes
        self.num_supernova = num_supernova

    def get_parameters(self):
        return ["Om", "alpha", "beta", "dscale", "dratio", "mean_MB",
                "mean_x1", "mean_c", "sigma_MB", "sigma_x1", "sigma_c",
                "calibration", "intrinsic_correlation"]

    def get_labels(self):
        mapping = OrderedDict([
            ("Om", r"$\Omega_m$"),
            ("w", r"$w$"),
            ("alpha", r"$\alpha$"),
            ("beta", r"$\beta$"),
            ("mean_MB", r"$\langle M_B \rangle$"),
            ("mean_x1", r"$\langle x_1^{%d} \rangle$"),
            ("mean_c", r"$\langle c^{%d} \rangle$"),
            ("sigma_MB", r"$\sigma_{\rm m_B}$"),
            ("sigma_x1", r"$\sigma_{x_1}$"),
            ("sigma_c", r"$\sigma_c$"),
            ("dscale", r"$\delta(0)$"),
            ("dratio", r"$\delta(\infty)/\delta(0)$"),
            ("calibration", r"$\delta \mathcal{Z}_%d$")
        ])
        return mapping

    def get_init(self):
        randoms = {
            "Om": uniform(0.1, 0.6),
            "alpha": uniform(-0.1, 0.4),
            "beta": uniform(0.1, 4.5),
            "dscale": uniform(-0.2, 0.2),
            "dratio": uniform(0, 1),
            "mean_MB": uniform(-20, -18),
            "mean_x1": uniform(-0.5, 0.5, size=self.num_redshift_nodes),
            "mean_c": uniform(-0.2, 0.2, size=self.num_redshift_nodes),
            "log_sigma_MB": uniform(-3, 1),
            "log_sigma_x1": uniform(-3, 1),
            "log_sigma_c": uniform(-3, 1),
            "deviations": normal(scale=0.2, size=(self.num_supernova, 3)),
            "calibration": uniform(-0.3, 0.3, size=8)
        }
        chol = [[1.0, 0.0, 0.0],
                [np.random.random() * 0.1 - 0.05, np.random.random() * 0.1 + 0.7, 0.0],
                [np.random.random() * 0.1 - 0.05, np.random.random() * 0.1 - 0.05,
                 np.random.random() * 0.1 + 0.7]]
        randoms["intrinsic_correlation"] = chol
        self.logger.info("Initial starting point is: %s" % randoms)
        return randoms

    def get_data(self, simulation, cosmology_index, add_zs=None):

        n_sne = self.num_supernova
        data = simulation.get_passed_supernova(n_sne, simulation=False, cosmology_index=cosmology_index)

        # Redshift shenanigans below used to create simpsons rule arrays
        # and then extract the right redshfit indexes from them
        redshifts = data["redshifts"]
        masses = data["masses"]

        n_z = 2000  # Defines how many points we get in simpsons rule

        num_nodes = self.num_redshift_nodes

        if num_nodes == 1:
            node_weights = np.array([[1]] * n_sne)
            nodes = [0.0]
        else:
            nodes = np.linspace(redshifts.min(), redshifts.max(), num_nodes)
            node_weights = self.get_node_weights(nodes, redshifts)

        if add_zs is not None:
            sim_data = add_zs(simulation)
            n_sim = sim_data["n_sim"]
            sim_redshifts = sim_data["sim_redshifts"]
            dz = max(redshifts.max(), sim_redshifts.max()) / n_z
            zs = sorted(redshifts.tolist() + sim_redshifts.tolist())
        else:
            sim_data = {}
            dz = redshifts.max() / n_z
            zs = sorted(redshifts.tolist())
            n_sim = 0

        added_zs = [0]
        pz = 0
        for z in zs:
            est_point = int((z - pz) / dz)
            if est_point % 2 == 0:
                est_point += 1
            est_point = max(3, est_point)
            new_points = np.linspace(pz, z, est_point)[1:-1].tolist()
            added_zs += new_points
            pz = z

        n_z = n_sne + n_sim + len(added_zs)
        n_simps = int((n_z + 1) / 2)
        to_sort = [(z, -1, -1) for z in added_zs] + [(z, i, -1) for i, z in enumerate(redshifts)]
        if add_zs is not None:
            to_sort += [(z, -1, i) for i, z in enumerate(sim_redshifts)]
        to_sort.sort()
        final_redshifts = np.array([z[0] for z in to_sort])
        sorted_vals = [(z[1], i) for i, z in enumerate(to_sort) if z[1] != -1]
        sorted_vals.sort()
        final = [int(z[1] / 2 + 1) for z in sorted_vals]
        # End redshift shenanigans

        update = {
            "n_z": n_z,
            "n_simps": n_simps,
            "zs": final_redshifts,
            "zspo": 1 + final_redshifts,
            "zsom": (1 + final_redshifts) ** 3,
            "redshift_indexes": final,
            "redshift_pre_comp": 0.9 + np.power(10, 0.95 * redshifts),
            "calib_std": np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1]),
            "num_nodes": num_nodes,
            "node_weights": node_weights,
            "nodes": nodes
        }

        if add_zs is not None:
            sim_sorted_vals = [(z[2], i) for i, z in enumerate(to_sort) if z[2] != -1]
            sim_sorted_vals.sort()
            sim_final = [int(z[1] / 2 + 1) for z in sim_sorted_vals]
            update["sim_redshift_indexes"] = sim_final
            update["sim_redshift_pre_comp"] = 0.9 + np.power(10, 0.95 * sim_redshifts)

            if num_nodes == 1:
                update["sim_node_weights"] = np.array([[1]] * sim_redshifts.size)
            else:
                update["sim_node_weights"] = self.get_node_weights(nodes, sim_redshifts)

        obs_data = np.array(data["obs_mBx1c"])
        self.logger.debug("Obs x1 std is %f, colour std is %f" % (np.std(obs_data[:, 1]), np.std(obs_data[:, 2])))

        # Add in data for the approximate selection efficiency in mB
        mean, std, alpha = simulation.get_approximate_correction()
        update["mB_mean"] = mean
        update["mB_width2"] = std**2
        update["mB_alpha2"] = alpha**2

        if np.all(masses == 0):
            update["mean_weight"] = 0
        else:
            update["mean_weight"] = 0.5

        final_dict = {**data, **update, **sim_data}
        return final_dict

    def get_node_weights(self, nodes, redshifts):
        indexes = np.arange(nodes.size)
        interps = interp1d(nodes, indexes, kind='linear', fill_value="extrapolate")(redshifts)
        node_weights = np.array([1 - np.abs(v - indexes) for v in interps])
        node_weights *= (node_weights <= 1) & (node_weights >= 0)
        node_weights = np.abs(node_weights)
        reweight = np.sum(node_weights, axis=1)
        node_weights = (node_weights.T / reweight).T
        return node_weights

    def correct_chain(self, dictionary, simulation, data):
        del dictionary["intrinsic_correlation"]
        return dictionary