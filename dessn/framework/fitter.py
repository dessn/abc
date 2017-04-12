import logging
import os
import pickle
import socket
from collections import OrderedDict

import numpy as np
import sys

from dessn.utility.doJob import write_jobscript_slurm


class Fitter(object):
    def __init__(self, temp_dir):
        self.models = []
        self.simulations = []
        self.num_cosmologies = 15
        self.num_walkers = 5
        self.num_cpu = self.num_cosmologies * self.num_walkers
        self.logger = logging.getLogger(__name__)
        self.temp_dir = temp_dir
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    def set_models(self, *models):
        self.models = models
        return self

    def set_simulations(self, *simulations):
        self.simulations = simulations
        return self

    def set_num_cosmologies(self, num_cosmologies):
        self.num_cosmologies = num_cosmologies
        return self

    def set_num_cpu(self, num_cpu=None):
        if num_cpu is None:
            self.num_cpu = self.num_cosmologies * self.num_walkers
        else:
            self.num_cpu = num_cpu

    def set_num_walkers(self, num_walkers):
        self.num_walkers = num_walkers
        return self

    def get_num_jobs(self):
        num_jobs = len(self.models) * len(self.simulations) * self.num_cosmologies * self.num_walkers
        return num_jobs

    def get_indexes_from_index(self, index):
        num_simulations = len(self.simulations)
        num_cosmo = self.num_cosmologies
        num_walkers = self.num_walkers

        num_per_model_sim = num_cosmo * num_walkers
        num_per_model = num_simulations * num_per_model_sim

        model_index = index // num_per_model
        index -= model_index * num_per_model
        sim_index = index // num_per_model_sim
        index -= sim_index * num_per_model_sim
        cosmo_index = index // num_walkers
        walker_index = index % num_walkers

        return model_index, sim_index, cosmo_index, walker_index

    def run_fit(self, model_index, simulation_index, cosmo_index, walker_index, num_cores=1):
        model = self.models[model_index]
        sim = self.simulations[simulation_index]

        out_file = self.temp_dir + "/stan_%d_%d_%d_%d.pkl" % (model_index, simulation_index, cosmo_index, walker_index)

        if num_cores == 1:
            w, n = 1000, 3000
        else:
            w, n = 300, 600

        import pystan
        data = model.get_data(sim, cosmo_index)
        self.logger.info("Running Stan job, saving to %s" % out_file)
        sm = pystan.StanModel(file=model.get_stan_file(), model_name="Cosmology")
        fit = sm.sampling(data=data, iter=n, warmup=w, chains=num_cores, init=model.get_init)
        self.logger.info("Stan finished sampling")

        # Get parameters
        params = [p for p in model.get_parameters() if p in fit.sim["pars_oi"]]
        if "weight" in fit.sim["pars_oi"]:
            self.logger.debug("Found weight to save")
            params.append("weight")
        if "posterior" in fit.sim["pars_oi"]:
            self.logger.debug("Found target to save")
            params.append("posterior")
        dictionary = fit.extract(pars=params)

        # Turn log scale parameters into normal scale to see them easier
        for key in list(dictionary.keys()):
            if key.find("log_") == 0:
                dictionary[key[4:]] = np.exp(dictionary[key])
                del dictionary[key]

        # Correct the chains if there is a weight function
        dictionary = model.correct_chain(dictionary, sim)

        with open(out_file, 'wb') as output:
            pickle.dump(dictionary, output)
        self.logger.info("Saved chain to %s" % out_file)

    def is_laptop(self):
        return "science" in socket.gethostname()

    def fit(self, file):

        num_jobs = self.get_num_jobs()
        num_models = len(self.models)
        num_simulations = len(self.models)
        self.logger.info("With %d models, %d simulations, %d cosmologies and %d walkers, have %d jobs" %
                         (num_models, num_simulations, self.num_cosmologies, self.num_walkers, num_jobs))

        if self.is_laptop():
            self.logger.info("Running Stan locally with 4 cores.")
            self.run_fit(0, 0, 0, 0, num_cores=4)
        else:
            if len(sys.argv) == 1:
                h = socket.gethostname()
                partition = "regular" if "edison" in h else "smp"
                filename = write_jobscript_slurm(file, name=os.path.basename(file),
                                                 num_tasks=self.get_num_jobs(), num_cpu=self.num_cpu,
                                                 delete=True, partition=partition)
                self.logger.info("Running batch job at %s" % filename)
                os.system("sbatch %s" % filename)
            else:
                index = int(sys.argv[1])
                mi, si, ci, wi = self.get_indexes_from_index(index)
                self.logger.info("Running model %d, sim %d, cosmology %d, walker number %d" % (mi, si, ci, wi))
                self.run_fit(mi, si, ci, wi)

    def load_file(self, filename):
        with open(filename, 'rb') as output:
            chain = pickle.load(output)
        self.logger.debug("Loaded pickle from %s" % filename)
        return chain

    def get_result_from_chain(self, chain, simulation_index, model_index):
        truth = self.simulations[simulation_index].get_truth_values_dict()
        mapping = self.models[model_index].get_labels()

        weight = chain.get("weight")
        old_weight = chain.get("old_weight")
        posterior = chain.get("posterior")

        parameters = list(mapping.keys())
        truth = {mapping[k]: truth.get(k) for k in mapping if k in truth.keys()}
        temp_list = []
        for p in parameters:
            vals = chain.get(p)
            if vals is None:
                continue
            if len(vals.shape) > 1:
                for i in range(vals.shape[1]):
                    temp_list.append((mapping[p] % i, vals[:, i]))
                    truth[mapping[p] % i] = truth[mapping[p]][i]
                del truth[mapping[p]]
            else:
                temp_list.append((mapping[p], vals))

        result = OrderedDict(temp_list)

        return result, truth, weight, old_weight, posterior

    def load(self, split_models=True, split_sims=True, split_cosmo=False):
        files = sorted([f for f in os.listdir(self.temp_dir) if f.endswith(".pkl")])
        filenames = [self.temp_dir + "/" + f for f in files]
        model_indexes = [int(f.split("_")[1]) for f in files]
        sim_indexes = [int(f.split("_")[2]) for f in files]
        cosmo_indexes = [int(f.split("_")[3]) for f in files]
        chains = [self.load_file(f) for f in filenames]

        results = []
        prev_model, prev_sim, prev_cosmo = None, None, None
        stacked = None
        for c, mi, si, ci in zip(chains, model_indexes, sim_indexes, cosmo_indexes):
            if (prev_cosmo != ci and split_cosmo) or (prev_model != mi and split_models) or (prev_sim != si and split_sims):
                if stacked is not None:
                    results.append(self.get_result_from_chain(stacked, si, mi))
                stacked = None
                prev_model = mi
                prev_sim = si
                prev_cosmo = ci
            if stacked is None:
                stacked = c
            else:
                for key in list(c.keys()):
                    stacked[key] = np.concatenate((stacked[key], c[key]))
        results.append(self.get_result_from_chain(stacked, si, mi))

        if len(results) == 1:
            return results[0]
        return results
