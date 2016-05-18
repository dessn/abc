from dessn.framework.samplers.scaffold import GenericSampler
import logging
import os
import numpy as np


class PolyChord(GenericSampler):  # pragma: no cover
    # No test coverage until I can figure out how to properly install PolyChord on travis
    def __init__(self, temp_dir, boost=0.0, num_repeats=None):
        """ Uses `PolyChord
        <https://ccpforge.cse.rl.ac.uk/gf/project/polychord/>`_ to fit the supplied
        model.

        Parameters
        ----------
        temp_dir : str
            Specifies the directory in which to save results
        boost : float, optional
            How much to boost the number of posterior samples by. Cannot exceed
            ``num_repeats``, which is by default ``5*num_dims``
        num_repeats : int, optional
            The number of slice slice-sampling steps to generate a new point.
        """

        self.logger = logging.getLogger(__name__)
        import PyPolyChord
        self.chain = None
        self.pool = None
        self.master = True
        self.temp_dir = os.path.abspath(temp_dir)
        self.boost = boost
        self.num_repeats = num_repeats

    def fit(self, model):
        """ Runs the sampler over the model and returns the flat chain of results

        Returns
        -------
        ndarray
            The final flattened chain of dimensions
            ``(num_dimensions, num_walkers * (num_steps - num_burn))``
        """
        import PyPolyChord

        num_dim = len(model._theta_names)
        if self.num_repeats is None:
            self.num_repeats = 2 * num_dim
        self.logger.debug("Fitting framework with %d dimensions" % num_dim)

        self.logger.info("Using PolyChord")
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)
        chain_dir = self.temp_dir + os.sep + "clusters"
        if not os.path.exists(chain_dir):
            os.mkdir(chain_dir)
        print("Do it")
        PyPolyChord.run_nested_sampling(model.get_log_posterior_polychord, num_dim, 0,
                                        base_dir=self.temp_dir,
                                        prior=model.get_hypercube_convert,
                                        file_root="chain",
                                        num_repeats=self.num_repeats,
                                        boost_posterior=self.boost)

        chain = np.loadtxt(self.temp_dir + os.sep + "chain_equal_weights.txt")
        self.logger.debug("Fit finished")
        return chain[:, 2:]
