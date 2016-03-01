import abc
from dessn.model.node import Node
from dessn.model.nodes.typeProb import Types


class Supernova(Node):
    """ Abstract supernova which others must implement
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def type(self):
        return


class SupernovaIa(object):
    """Models a type Ia supernova

    Models the luminosity distribution of type Ia supernovas statically (without
    internal parameters).

    .. math::
        P(L) \sim N(

    Parameters
    ----------
    log_luminosity : float
        Represented by the variable :math:`\mu`
    sigma_luminosity : float
        Represented by the variable :math:`\sigma`
    """
    def __init__(self, log_luminosity, sigma_luminosity):
        self.log_luminosity = log_luminosity
        self.sigma_luminosity = sigma_luminosity

    def get_luminosity_prob(self, log_luminosity):
        return

    def type(self):
        return Types.snIa


