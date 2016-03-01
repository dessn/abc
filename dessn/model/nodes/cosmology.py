import abc
from dessn.model.node import Node


class Cosmology(Node):
    __metaclass__ = abc.ABCMeta

    def get_name(self):
        return "Cosmology"


class FlatWCDM(Cosmology):
    def __call__(self):
        pass
