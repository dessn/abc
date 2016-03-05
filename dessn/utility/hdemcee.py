from time import time
import logging
import numpy as np


class EmceeWrapper(object):
    def __init__(self, sampler, filename=None):
        self.sampler = sampler
        self.filename = filename
        self.logger = logging.getLogger(__name__)
        self.chain = None

    def run_chain(self, num_steps, num_burn, num_walkers, num_dim, start=None, save_interval=300, save_dim=None):
        assert num_steps > num_burn, "num_steps has to be larger than num_burn"
        if save_dim is not None:
            assert save_dim <= num_dim, "You cannot save more dimensions than you actually have"
        else:
            save_dim = num_dim

        past_chain = None
        pos = None
        if self.filename is not None:
            chain_file = self.filename + ".chain.npy"
            position_file = self.filename + ".pos.npy"
            try:
                pos = np.load(position_file)
                print(pos)
                past_chain = np.load(chain_file)
                self.logger.info("Found chain of %d steps" % past_chain.shape[1])
            except IOError:
                self.logger.info("Prior chain and/or does not exist")

        if start is None and pos is None:
            raise ValueError("You need to have either a starting position or existing chains")

        if pos is None:
            pos = start

        step = 0
        self.chain = np.zeros((num_walkers, num_steps, save_dim))
        if past_chain is not None and past_chain.shape[1] <= num_steps:
            step = past_chain.shape[1]
            print(step)
            num = num_steps - step
            print(num)
            self.chain[:, :step, :] = past_chain
        else:
            num = num_steps

        t = time()
        for result in self.sampler.sample(pos, iterations=num, storechain=False):
            self.chain[:, step, :] = result[0][:, :save_dim]
            step += 1
            if self.filename is not None:
                t2 = time()
                if t2 - t > save_interval or step == num_steps:
                    t = t2
                    position = result[0]
                    np.save(position_file, position)
                    np.save(chain_file, self.chain[:, :step, :])
                    self.logger.debug("Saving chain with %d steps" % step)
        return self.get_results()

    def get_results(self):
        return self.chain.reshape((-1, self.chain.shape[2]))
