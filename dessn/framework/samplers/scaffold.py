import abc


class GenericSampler(object):
    __metaclass__ = abc.ABCMeta

    def can_use(self):
        """ Returns a boolean expression if the sampler is available or not. """
        raise NotImplementedError()

    def fit(self, model):
        raise NotImplementedError()
