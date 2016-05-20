import abc


class GenericSampler(object):
    __metaclass__ = abc.ABCMeta

    def fit(self, model):
        """ Fits a given model using the Sampler.

        Parameters
        ----------
        model : :py:mod:`Model`_
            The model with which to fit

        Returns
        -------
        ``np.ndarray``
            The chains from the fit, ready to be consumed with :py:mod`ChainConsumer`_
        """
        raise NotImplementedError()
