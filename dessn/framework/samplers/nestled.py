from dessn.framework.samplers.scaffold import GenericSampler
import numpy as np
import os
import logging


class NestledSampler(GenericSampler):
    def __init__(self, temp_dir=None):
        import nestle
        self.logger = logging.getLogger(__name__)
        self.temp_dir = temp_dir
        if temp_dir is not None and not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    def fit(self, kwargs):
        """ Runs the sampler over the model and returns the flat chain of results

        Parameters
        ----------
        kwargs : dict
            Containing the following information at a minimum:

            - log_posterior : function
                A function which takes a list of parameters and returns
                the log posterior and derived parameters (a tuple)
            - hypercube : function
                A function that takes a list of parameters in the range 0 to 1
                and scales them to the correct ranges for the model
            - num_dim : int
                The number of dimensions in the model
            - uid : str, optional
                The UID with which to save results
        Returns
        -------
        dict
            A dictionary with key "chains" containing the final
            flattened chain of dimensions
             ``(num_dimensions, num_walkers * (num_steps - num_burn))``
        """
        log_posterior_polychord = kwargs["log_posterior"]
        hypercube = kwargs["hypercube"]
        num_dim = kwargs["num_dim"]
        uid = kwargs.get("uid")
        if uid is None:
            uid = "nestle"
        assert log_posterior_polychord is not None
        assert hypercube is not None
        assert num_dim is not None

        if self.temp_dir is not None:
            filename = self.temp_dir + os.sep + uid + "_nestle.npy"
            if os.path.exists(filename):
                result = np.load(filename)
                self.logger.info("Returning saved results")
                return {"chain": result[:, :-1], "weights": result[:, -1]}
        else:
            filename = None

        import nestle
        self.logger.info("Starting fit")
        res = nestle.sample(log_posterior_polychord, hypercube, num_dim,
                            npoints=max(100, 5 * num_dim), callback=nestle.print_progress)
        if filename is not None:
            self.logger.info("Saving results")
            np.save(filename, np.hstack((res.samples, res.weights)))
        self.logger.info("Returning fit results")
        return {"chain": res.samples, "weights": res.weights}
