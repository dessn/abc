import abc


class GenericSampler(object):
    __metaclass__ = abc.ABCMeta

    def fit(self, kwargs):
        """ Fits a given model using the Sampler.

        Parameters
        ----------
        kwargs : dict
            Dictionary of keyword arguments utilised by the fitters

        Returns
        -------
        dict
            A dictionary of results containing:
                - *chain*: the chain
                - *weights*: chain weights if applicable
        """
        raise NotImplementedError()
