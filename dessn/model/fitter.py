import logging


class Fitter(object):
    def __init__(self):
        self.models = []
        self.simulations = []
        self.num_cosmologies = 15
        self.num_walkers = 5
        self.logger = logging.getLogger(__name__)

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

    def fit(self, index=None):
        num_jobs = self.get_num_jobs()
        self.logger.info("With %d models, %d simulations, %d cosmologies and %d walkers, have %d jobs" %
                         (len(self.models), len(self.simulations), self.num_cosmologies, self.num_walkers, num_jobs))

        # Need to think about the interplay between a scheduling method and this class.
        # How much responsibility should fitter take on?
