from dessn.framework.samplers.metropolisHastings import MetropolisHastings
from dessn.framework.samplers.scaffold import GenericSampler
import os
import numpy as np
import logging
from joblib import Parallel, delayed


class BatchMetropolisHastings(GenericSampler):
    """ An ensemble of self tuning Metropolis Hastings Sampler

    Parameters
    ----------
    num_walkers : int, optional
        The number of walkers to use
    temp_dir : str, optional
        The temporary directory to save results
    sampler : GenericSampler
        Which simpler to use
    kwargs : dict, optional
        The parameters to pass to the sampler on init
    num_cores : int, optional
        The number of cores to use.
    """
    def __init__(self, num_walkers=8, temp_dir=None, sampler=MetropolisHastings, kwargs=None,
                 num_cores=4):
        self.logger = logging.getLogger(__name__)
        self.num_walkers = num_walkers
        self.temp_dir = temp_dir
        self.sampler = sampler
        self.kwargs = kwargs
        self.num_cores = num_cores
        self.final = None
        if temp_dir is None:
            self.walker_temp_dirs = [None for i in range(num_walkers)]
        else:
            self.walker_temp_dirs = [temp_dir + os.sep + "%d" % i for i in range(num_walkers)]

        if self.kwargs is None:
            self.kwargs = {}

    def fit_all(self, kwargs):
        self.logger.info("Starting fit")
        res = Parallel(n_jobs=self.num_cores, max_nbytes="20M", verbose=100)(delayed(fit_sampler)(
            self.sampler, self.kwargs, self.walker_temp_dirs[i], kwargs)
                                              for i in range(self.num_walkers))
        return res

    def fit(self, kwargs):
        res = self.fit_all(kwargs)
        chain = np.vstack(tuple([r["chain"] for r in res]))
        result = {"chain": chain}
        if res[0].get("weights") is not None:
            result["weights"] = np.concatenate(tuple([r["weights"] for r in res]))
        if res[0].get("posterior") is not None:
            result["posterior"] = np.concatenate(tuple([r["posterior"] for r in res]))
        self.logger.debug("Returning fit results")
        return result


def fit_sampler(sampler, sampler_kwargs, temp_dir, fit_kwargs):
    sampler_kwargs["temp_dir"] = temp_dir
    sampler = sampler(**sampler_kwargs)
    return sampler.fit(fit_kwargs)
