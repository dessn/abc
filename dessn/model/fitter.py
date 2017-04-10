import logging
import os
import pickle

import numpy as np


class Fitter(object):
    def __init__(self, temp_dir):
        self.models = []
        self.simulations = []
        self.num_cosmologies = 15
        self.num_walkers = 5
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

    def run_fit_laptop(self):
        self.run_fit(0, 0, 0, 0, num_cores=4)

    def run_fit(self, model_index, simulation_index, cosmo_index, walker_index, num_cores=1):
        model = self.models[model_index]
        sim = self.simulations[simulation_index]

        out_file = self.temp_dir + "/stan_%d_%d_%d_%d.pkl" % (model_index, simulation_index, cosmo_index, walker_index)

        if num_cores == 1:
            w, n = 1000, 3000
        else:
            w, n = 300, 600

        import pystan

        self.logger.info("Running Stan job, saving to %s" % out_file)
        sm = pystan.StanModel(file=model.get_stan_file(), model_name="Cosmology")
        fit = sm.sampling(data=model.get_data(), iter=n, warmup=w, chains=num_cores, init=model.get_init)
        self.logger.info("Stan finished sampling")

        # Get parameters
        params = [p for p in model.get_parameters() if p in fit.sim["pars_oi"]]
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

    def fit(self, index=None):
        num_jobs = self.get_num_jobs()
        num_models = len(self.models)
        num_simulations = len(self.models)
        self.logger.info("With %d models, %d simulations, %d cosmologies and %d walkers, have %d jobs" %
                         (num_models, num_simulations, self.num_cosmologies, self.num_walkers, num_jobs))

        if index is None:
            self.logger.info("Running Stan locally with 4 cores.")
        else:
            mi, si, ci, wi = self.get_indexes_from_index(index)
            self.logger.info("Running model %d, sim %d, cosmology %d, walker number %d" % (mi, si, ci, wi))


        # Need to think about the interplay between a scheduling method and this class.
        # How much responsibility should fitter take on?
