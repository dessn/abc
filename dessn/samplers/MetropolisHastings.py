from dessn.samplers.GenericSampler import Sampler
import os
import numpy as np
from time import time
import logging


class MetropolisHastings(Sampler):
    """ Self tuning Metropolis Hastings Sampler

    Parameters
    ----------
    log_posterior : function
        A function that, when given an array of parameters, returns the log posterior of the model.
    start : function/list[numbers]/np.ndarray
        The starting position. Either a list/array of numbers, of a function that
        returns one
    uid : str, optional
        A UID string used when saving results. Useful if you want to run many samplers
        in the same temp directory.
    num_burn : int, optional
        The number of burn in steps. TODO: Tune this automatically
    num_steps : int, optional
        The number of steps to take after burn in. TODO: Tune this automatically
    sigma_adjust : int, optional
        During the burn in, how many steps between adjustment to the step size
    covariance_adjust : int, optional
        During the burn in, how many steps between adjustment to the parameter covariance.
    temp_dir : str, optional
        The location of a folder to save the results, such as the last position and chain
    save_dims : int, optional
        If given, the final chain will contain only the first ``save_dim`` parameters.
        Useful for discarding many nuisance parameters that would bloat the size of the chain.
    accept_ratio : float, optional
        The desired acceptance ratio
    """
    def __init__(self, log_posterior, start, uid="mh", num_burn=10000, num_steps=10000,
                 sigma_adjust=100, covariance_adjust=1000, temp_dir=None,
                 save_dims=None, save_interval=300,
                 accept_ratio=0.4):
        super().__init__(temp_dir)
        self.logger = logging.getLogger(__name__)
        self.log_posterior = log_posterior
        self.start = start
        self.num_burn = num_burn
        self.num_steps = num_steps
        self.sigma_adjust = sigma_adjust
        self.covariance_adjust = covariance_adjust
        self.save_dims = save_dims
        self.save_interval = save_interval
        self.accept_ratio = accept_ratio
        self._do_save = temp_dir is not None and save_interval is not None
        self.space = 3  # log posterior, sigma, weight
        self.IND_P = 0
        self.IND_S = 1
        self.IND_W = 2

        if temp_dir is None:
            self.position_file = None
            self.burn_file = None
            self.chain_file = None
            self.covariance_file = None
            self.sigma_file = None
        else:
            self.position_file = temp_dir + os.sep + "%s_position.npy" % uid
            self.burn_file = temp_dir + os.sep + "%s_burn.npy" % uid
            self.chain_file = temp_dir + os.sep + "%s_chain.npy" % uid
            self.covariance_file = temp_dir + os.sep + "%s_covariance.npy" % uid
            self.sigma_file = temp_dir + os.sep + "%s_sigma.npy" % uid

    def fit(self):
        """ Runs the fit """
        position, burnin, chain, covariance, sigma = self._load()
        position = self._ensure_position(position)
        sigma = self._ensure_sigma(sigma, position)
        if chain is not None and burnin.shape[0] == self.num_burn:
            res = self._do_chain(position, sigma, covariance, chain=chain)
        else:
            position, covariance, sigma = self._do_burnin(position, burnin, sigma, covariance)
            res = self._do_chain(position, sigma, covariance)

        return res

    def _do_chain(self, position, sigma, covariance, chain=None):
        dims = self.save_dims if self.save_dims is not None else position.size - self.space
        size = dims + self.space
        if chain is None:
            current_step = 1
            chain = np.zeros((self.num_steps, size))
            chain[0, :] = position[:size]
        elif chain.shape[0] < self.num_steps:
            current_step = chain.shape[0]
            chain = np.vstack((chain, np.zeros((self.num_steps - chain.shape[0], size))))
        else:
            current_step = self.num_steps

        last_save_time = time()
        self.logger.debug("Starting chain")
        while current_step < self.num_steps:
            position, weight = self._get_next_step(position, sigma, covariance)
            chain[current_step - 1, self.IND_W] += 1
            chain[current_step, :] = position[:size]
            current_step += 1
            if current_step == self.num_steps or \
                    (self._do_save and time() - last_save_time > self.save_interval):
                self._save(position, None, chain[:current_step, :], None, None)

        return chain[:, self.space:], chain[:, self.IND_W]

    def _do_burnin(self, position, burnin, sigma, covariance):
        if burnin is None:
            # Initialise burning to all zeros. 2 from posterior and step size
            burnin = np.zeros((self.num_burn, position.size))
            current_step = 1
            burnin[0, :] = position
        elif burnin.shape[0] < self.num_burn:
            current_step = burnin.shape[0]
            # If we only saved part of the burnin to save size, add the rest in as zeros
            burnin = np.vstack((burnin, np.zeros((self.num_burn - burnin.shape[0], position.size))))
        else:
            current_step = self.num_burn
        if covariance is None:
            covariance = np.identity(position.size - self.space)

        last_save_time = time()
        self.logger.debug("Starting burn in")
        while current_step < self.num_burn:
            # If sigma adjust, adjust
            if current_step % self.sigma_adjust == 0 and current_step > 0:
                burnin[current_step, self.IND_S] = self._adjust_sigma_ratio(burnin, current_step)
            # If covariance adjust, adjust
            if current_step % self.covariance_adjust == 0 and current_step > 0:
                sigma, covariance = self._adjust_covariance(burnin, current_step)

            # Get next step
            burnin[current_step, :], weight = self._get_next_step(burnin[current_step - 1, :],
                                                          sigma, covariance, burnin=True)
            burnin[current_step - 1, self.IND_W] = weight
            current_step += 1
            if current_step == self.num_burn or \
                    (self._do_save and time() - last_save_time > self.save_interval):
                self._save(burnin[current_step - 1, :], burnin[:current_step, :],
                           None, covariance, sigma)
                last_save_time = time()

        return burnin[-1, :], covariance, sigma

    def _ensure_position(self, position):
        """ Ensures that the position object, which can be none from loading, is a
        valid [starting] position.
        """
        if position is None:
            if callable(self.start):
                position = self.start()
            else:
                position = self.start
            if type(position) == list:
                position = np.array(position)
            position = np.concatenate(([self.log_posterior(position), 1, 1], position))
            # Starting log posterior is infinitely unlikely, sigma size of 1 to begin with
        return position

    def _ensure_sigma(self, sigma, position):
        if sigma is None:
            return np.ones(position.size - self.space)
        return sigma

    def _adjust_sigma_ratio(self, burnin, index):
        subsection = burnin[index - self.sigma_adjust:index, :]
        actual_ratio = 1 / np.average(subsection[:, self.IND_W])

        sigma_ratio = burnin[index - 1, self.IND_S]
        if actual_ratio < self.accept_ratio:
            sigma_ratio *= 0.9  # TODO: Improve for high dimensionality
        else:
            sigma_ratio /= 0.9
        self.logger.debug("Adjusting sigma: Want %0.2f, got %0.2f. "
                          "Updating ratio to %0.3f" % (self.accept_ratio, actual_ratio, sigma_ratio))
        burnin[index - 1, self.IND_S] = sigma_ratio

    def _adjust_covariance(self, burnin, index):
        subset = burnin[int(np.floor(index/2)):index, :]
        covariance = np.cov(subset[:, self.space:].T, fweights=subset[:, self.IND_W])
        evals, evecs = np.linalg.eig(covariance)
        sigma = np.sqrt(np.abs(evals)) * 2.3 / np.sqrt(evals.size)
        self.logger.debug("Adjusting covariance and reseting sigma ratio")
        return sigma, evecs

    def _propose_point(self, position, sigma, covariance):
        rotated_params = np.dot(position[self.space:], covariance)
        new_params = rotated_params + \
                     sigma * position[self.IND_S] * np.random.normal(size=sigma.size)
        return np.dot(covariance, new_params)

    def _get_next_step(self, position, sigma, covariance, burnin=False):
        attempts = 1
        past_pot = position[self.IND_P]
        while True:
            pot = self._propose_point(position, sigma, covariance)
            posterior = self.log_posterior(pot)

            # print(posterior, past_pot, posterior > past_pot, np.exp(posterior - past_pot))
            if posterior > past_pot or np.exp(posterior - past_pot) > np.random.uniform():
                result = np.concatenate(([posterior, position[self.IND_S], 1], pot))
                return result, attempts
            else:
                attempts += 1
                if attempts > 100 and burnin:
                    posterior[self.IND_S] *= 0.9

    def _load(self):
        position = None
        if os.path.exists(self.position_file):
            position = np.load(self.position_file)
        burnin = None
        if os.path.exists(self.burn_file):
            burnin = np.load(self.burn_file)
        chain = None
        if os.path.exists(self.chain_file):
            chain = np.load(self.chain_file)
        covariance = None
        if os.path.exists(self.covariance_file):
            covariance = np.load(self.covariance_file)
        sigma = None
        if os.path.exists(self.sigma_file):
            sigma = np.load(self.sigma_file)
        return position, burnin, chain, covariance, sigma

    def _save(self, position, burnin, chain, covariance, sigma):
        self.logger.debug("Serialising results to file")
        if position is not None:
            np.save(self.position_file, position)
        if burnin is not None:
            np.save(self.burn_file, burnin)
        if chain is not None:
            np.save(self.chain_file, chain)
        if covariance is not None:
            np.save(self.covariance_file, covariance)
        if sigma is not None:
            np.save(self.sigma_file, sigma)

