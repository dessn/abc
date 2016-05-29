from dessn.framework.samplers.scaffold import GenericSampler
import os
import numpy as np
from time import time
import logging


class MetropolisHastings(GenericSampler):
    """ Self tuning Metropolis Hastings Sampler

    Parameters
    ----------
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
    save_interval : int, optional
        How many seconds should pass between saving data snapshots
    accept_ratio : float, optional
        The desired acceptance ratio
    callback : function, optional
        If set, passes the log posterior, position and weight for each step in the burn
        in and the chain to the function. Useful for plotting the walks whilst the
        chain is running.
    """
    def __init__(self, num_burn=10000, num_steps=10000,
                 sigma_adjust=100, covariance_adjust=1000, temp_dir=None,
                 save_interval=300, accept_ratio=0.234, callback=None):
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        self.logger = logging.getLogger(__name__)
        self.log_posterior = None
        self.start = None
        self.save_dims = None
        self.callback = callback

        self.num_burn = num_burn
        self.num_steps = num_steps
        self.sigma_adjust = sigma_adjust # Also should be at least 5 x num_dim
        self.covariance_adjust = covariance_adjust
        self.save_interval = save_interval
        self.accept_ratio = accept_ratio
        self._do_save = temp_dir is not None and save_interval is not None
        self.space = 3  # log posterior, sigma, weight
        self.IND_P = 0
        self.IND_S = 1
        self.IND_W = 2

        self.position_file = None
        self.burn_file = None
        self.chain_file = None
        self.covariance_file = None

    def fit(self, kwargs):
        """
        Fit the model

        Parameters
        ----------
        kwargs : dict
            Containing the following information at a minimum:

            - log_posterior : function
                A function that takes a list of parameters and returns the
                log posterior
            - start : function|list|ndarray
                Either a starting position, or a function that can be called
                to generate a starting position
            - save_dims : int, optional
                Only return values for the first ``save_dims`` parameters.
                Useful to remove numerous marginalisation parameters if running
                low on memory or hard drive space.
            - uid : str, optional
                A unique identifier used to differentiate different fits
                if two fits both serialise their chains and use the
                same temporary directory

        Returns
        -------
        dict
            A dictionary containing the chain and the weights
        """
        log_posterior = kwargs.get("log_posterior")
        start = kwargs.get("start")
        save_dims = kwargs.get("save_dims")
        uid = kwargs.get("uid")
        assert log_posterior is not None
        assert start is not None
        if uid is None:
            uid = "mh"
        self._update_temp_files(uid)
        self.save_dims = save_dims
        self.log_posterior = log_posterior
        self.start = start
        position, burnin, chain, covariance = self._load()
        if burnin is not None:
            self.logger.debug("Found burnin of size %d" % burnin.shape[0])
        if chain is not None:
            self.logger.debug("Found chain of size %d" % chain.shape[0])
        position = self._ensure_position(position)

        if chain is not None and burnin is not None and burnin.shape[0] == self.num_burn:
            c, w, p = self._do_chain(position, covariance, chain=chain)
        else:
            position, covariance = self._do_burnin(position, burnin, covariance)
            c, w, p = self._do_chain(position, covariance)
        self.logger.info("Returning results")
        return {"chain": c, "weights": w, "posterior": p}

    def _do_burnin(self, position, burnin, covariance):
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
        num_dim = position.size - self.space
        if covariance is None:
            covariance = np.identity(position.size - self.space)

        last_save_time = time()
        self.logger.info("Starting burn in")
        while current_step < self.num_burn:
            # If sigma adjust, adjust
            if current_step % self.sigma_adjust == 0:
                burnin[current_step, self.IND_S] = self._adjust_sigma_ratio(burnin, current_step)
            # If covariance adjust, adjust
            if current_step % self.covariance_adjust == 0 and current_step > 0 and current_step > num_dim * 5:
                covariance = self._adjust_covariance(burnin, current_step)

            # Get next step
            burnin[current_step, :], weight = self._get_next_step(burnin[current_step - 1, :],
                                                          covariance, burnin=True)
            burnin[current_step - 1, self.IND_W] = weight
            if self.callback is not None:
                self.callback(burnin[current_step - 1, self.IND_P],
                              burnin[current_step - 1, self.space:self.space + self.save_dims],
                              weight=burnin[current_step - 1, self.IND_W])
            current_step += 1
            if current_step == self.num_burn or \
                    (self._do_save and (time() - last_save_time) > self.save_interval):
                self._save(burnin[current_step - 1, :], burnin[:current_step, :],
                           None, covariance)
                last_save_time = time()

        return burnin[-1, :], covariance

    def _do_chain(self, position, covariance, chain=None):
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
        self.logger.info("Starting chain")
        while current_step < self.num_steps:
            position, weight = self._get_next_step(position, covariance)
            chain[current_step, :] = position[:size]
            chain[current_step - 1, self.IND_W] = weight
            if self.callback is not None:
                self.callback(chain[current_step - 1, self.IND_P], chain[current_step - 1, self.space:],
                              weight=chain[current_step - 1, self.IND_W])
            current_step += 1
            if current_step == self.num_steps or \
                    (self._do_save and (time() - last_save_time) > self.save_interval):
                self._save(position, None, chain[:current_step, :], None)
                last_save_time = time()
        return chain[:, self.space:], chain[:, self.IND_W], chain[:, self.IND_P]

    def _update_temp_files(self, uid):
        if self.temp_dir is not None:
            self.position_file = self.temp_dir + os.sep + "%s_mh_position.npy" % uid
            self.burn_file = self.temp_dir + os.sep + "%s_mh_burn.npy" % uid
            self.chain_file = self.temp_dir + os.sep + "%s_mh_chain.npy" % uid
            self.covariance_file = self.temp_dir + os.sep + "%s_mh_covariance.npy" % uid

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
            # Starting log posterior is infinitely unlikely, sigma ratio of 1 to begin with
        return position

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
        params = burnin.shape[1] - self.space
        if params == 1:
            return np.ones((1,1))
        subset = burnin[int(np.floor(index/2)):index, :]
        covariance = np.cov(subset[:, self.space:].T, fweights=subset[:, self.IND_W])
        res = np.linalg.cholesky(covariance)
        self.logger.debug("Adjusting covariance and resetting sigma ratio")
        return res

    def _propose_point(self, position, covariance):
        p = position[self.space:]
        eta = np.random.normal(size=p.size)
        step = np.dot(covariance, eta) * position[self.IND_S]
        return p + step

    def _get_next_step(self, position, covariance, burnin=False):
        attempts = 1
        counter = 1
        past_pot = position[self.IND_P]
        while True:
            pot = self._propose_point(position, covariance)
            posterior = self.log_posterior(pot)
            if posterior > past_pot or np.exp(posterior - past_pot) > np.random.uniform():
                result = np.concatenate(([posterior, position[self.IND_S], 1], pot))
                return result, attempts
            else:
                attempts += 1
                counter += 1
                if counter > 50 and burnin:
                    position[self.IND_S] *= 0.9
                    counter = 0

    def _load(self):
        position = None
        if self.position_file is not None and os.path.exists(self.position_file):
            position = np.load(self.position_file)
        burnin = None
        if self.burn_file is not None and os.path.exists(self.burn_file):
            burnin = np.load(self.burn_file)
        chain = None
        if self.chain_file is not None and os.path.exists(self.chain_file):
            chain = np.load(self.chain_file)
        covariance = None
        if self.covariance_file is not None and os.path.exists(self.covariance_file):
            covariance = np.load(self.covariance_file)
        return position, burnin, chain, covariance

    def _save(self, position, burnin, chain, covariance):
        if position is not None and self.position_file is not None:
            np.save(self.position_file, position)
        if burnin is not None and self.burn_file is not None:
            self.logger.info("Serialising results to file. Burnin has %d steps" % burnin.shape[0])
            np.save(self.burn_file, burnin)
        if chain is not None and self.chain_file is not None:
            self.logger.info("Serialising results to file. Chain has %d steps" % chain.shape[0])
            np.save(self.chain_file, chain)
        if covariance is not None and self.covariance_file is not None:
            np.save(self.covariance_file, covariance)


