from time import time
import logging
import numpy as np
from emcee import PTSampler


class EmceeWrapper(object):
    def __init__(self, sampler):
        self.sampler = sampler
        self.logger = logging.getLogger(__name__)
        self.chain = None

    def run_chain(self, num_steps, num_burn, num_walkers, num_dim, start=None, save_interval=300, save_dim=None, temp_dir=None):
        assert num_steps > num_burn, "num_steps has to be larger than num_burn"
        if save_dim is not None:
            assert save_dim <= num_dim, "You cannot save more dimensions than you actually have"
        else:
            save_dim = num_dim

        past_chain = None
        pos = None
        if temp_dir is not None:
            self.logger.debug("Looking in tempr dir %s" % temp_dir)
            chain_file = temp_dir + ".chain.npy"
            position_file = temp_dir + ".pos.npy"
            try:
                pos = np.load(position_file)
                past_chain = np.load(chain_file)
                self.logger.info("Found chain of %d steps" % past_chain.shape[1])
            except IOError:
                self.logger.info("Prior chain and/or does not exist. Looked in %s" % position_file)

        if start is None and pos is None:
            raise ValueError("You need to have either a starting function or existing chains")

        if pos is None:
            pos = start(num_walkers)

        step = 0
        self.chain = np.zeros((num_walkers, num_steps, save_dim))
        if past_chain is not None and past_chain.shape[1] <= num_steps:
            step = past_chain.shape[1]
            num = num_steps - step
            self.chain[:, :step, :] = past_chain
            self.logger.debug("A further %d steps are required" % num)
        else:
            num = num_steps
            self.logger.debug("Running full chain of %d steps" % num)

        t = time()
        self.logger.debug("Starting sampling. Saving to %s ever %d seconds" % (temp_dir, save_interval))
        for result in self.sampler.sample(pos, iterations=num, storechain=False):
            if isinstance(self.sampler, PTSampler):
                self.chain[:, step, :] = result[0][0, :, :save_dim]
            else:
                self.chain[:, step, :] = result[0][:, :save_dim]
            step += 1
            if temp_dir is not None and save_interval is not None:
                t2 = time()
                if t2 - t > save_interval or step == num_steps:
                    t = t2
                    position = result[0]
                    np.save(position_file, position)
                    np.save(chain_file, self.chain[:, :step, :])
                    self.logger.debug("Saving chain with %d steps" % step)
        return self.get_results(num_burn)

    def get_results(self, num_burn):
        return self.chain[:, num_burn:, :].reshape((-1, self.chain.shape[2]))
