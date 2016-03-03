import abc
from enum import Enum


class NodeType(Enum):
    UNDERLYING = 1
    OBSERVED = 2
    LATENT = 3
    TRANSFORMATION = 4


class Node(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, names, labels, parameter_type):
        assert type(names) == list or type(names) == str, "Supplied name %s is not a string or list" % names
        assert type(labels) == list or type(labels) == str, "Supplied label text %s is not a string or list" % labels
        assert type(parameter_type) == NodeType, "Supplied parameter_type should be an enum from ParameterType"
        if type(names) == str:
            self.names = [names]
        else:
            for name in names:
                assert type(name) == str, "Entry in list %s is not a string" % name
            self.names = names
        if type(labels) == str:
            self.labels = [labels]
        else:
            for label in labels:
                assert type(label) == str, "Entry in list %s is not a string" % labels
            self.labels = labels


class NodeObserved(Node):
    def __init__(self, names, labels, datas):
        if type(names) == list:
            assert len(names) == len(labels) and len(names) == len(datas)
        super(NodeObserved, self).__init__(names, labels, NodeType.OBSERVED)
        self.datas = datas

    def get_data(self):
        return {name: data for name, data in zip(self.names, self.datas)}


class NodeUnderlying(Node):
    def __init__(self, names, labels):
        super(NodeUnderlying, self).__init__(names, labels, NodeType.UNDERLYING)

    @abc.abstractmethod
    def get_log_prior(self, data):
        pass


class NodeTransformation(Node):
    def __init__(self, names, labels):
        super(NodeTransformation, self).__init__(names, labels, NodeType.TRANSFORMATION)

    @abc.abstractmethod
    def get_transformation(self):
        pass


class NodeLatent(Node):
    def __init__(self, names, labels):
        super(NodeLatent, self).__init__(names, labels, NodeType.LATENT)

    @abc.abstractmethod
    def get_num_latent(self):
        pass

