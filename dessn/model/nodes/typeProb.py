from dessn.model.node import Node
from enum import Enum
import abc


class Types(Enum):
    """Possible target types
    """
    snIa = 1
    snII = 2


# todo: Consider if we should have a discrete and continuous subclasses of Node

class TypeProbability(Node):
    """Abstract type probability node, from which all implementations should inherit
    """
    __metaclass__ = abc.ABCMeta

    def get_name(self):
        return "Type Probability"


class TypeProbabilitySimple(TypeProbability):
    """The Type probability node.

    Takes *some information* to determine the probability of
    of the object in question being a type of supernova.

    Parameters
    ----------
    relative_rate : Optional[str]
        Relative rate of SnIa / SnII


    """

    def __init__(self, relative_rate=0.333):
        super(TypeProbabilitySimple, self).__init__()
        self.relative_rate = relative_rate

    def __call__(self):
        return {Types.snIa: self.relative_rate, Types.snII: 1 - self.relative_rate}
