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
        num_models = len(self.models)
        num_simulations = len(self.models)
        self.logger.info("With %d models, %d simulations, %d cosmologies and %d walkers, have %d jobs" %
                         (num_models, num_simulations, self.num_cosmologies, self.num_walkers, num_jobs))

        if index is None:
            self.logger.info("Running Stan locally with 4 cores.")
        else:
            # Figure out which model / simulation / cosmology / walker we are on based on the index
            pass


        # Need to think about the interplay between a scheduling method and this class.
        # How much responsibility should fitter take on?
