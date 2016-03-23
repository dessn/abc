import abc

class Parameter(object):
    def __init__(self, name, label, group=None):
        self.name = name
        self.label = label
        self.group = group

class ObservedParameter(Parameter):
    def __init__(self, name, label, data, group=None):
        super(ObservedParameter, self).__init__(name, label, group=group)
        self.data = data

class DiscreteParameter(Parameter):
    def __init__(self, name, label, group=None):
        super(DiscreteParameter, self).__init__(name, label, group=group)

    @abc.abstractmethod
    def get_discrete(self, *args):
        pass
    @abc.abstractmethod
    def get_discrete_requirements(self):
        return

class UnderlyingParameter(Parameter):
    def __init__(self, name, label, group=None):
        super(UnderlyingParameter, self).__init__(name, label, group=group)
    @abc.abstractmethod
    def get_log_prior(self, *data):
        return

class LatentParameter(Parameter):
    def __init__(self, name, label, group=None):
        super(LatentParameter, self).__init__(name, label, group=group)

class TransformedParameter(Parameter):
    def __init__(self, name, label, group=None):
        super(TransformedParameter, self).__init__(name, label, group=group)


class Edge(object):
    def __init__(self, probability_of, given):
        self.probability_of = probability_of
        self.given = given
    @abc.abstractmethod
    def get_log_likelihood(self, *data):
        return

class Dataset(object):
    def __init__(self, n):
        self.n = n
        self.observed = []

    def add_observable(self, parameter):
        assert type(parameter.data) == list and len(parameter.data) == self.n, \
            "Your parameter needs to have data of a python list of of %d values" % self.n
        self.observed.append(parameter)


class Model(object):
    def __init__(self):
        self.datasets = []
        self.parameters = []
        self.edges = []

    def add_dataset(self, dataset):
        self.datasets.append(dataset)

    def add_parameter(self, p):
        self.parameters.append(p)

    def add_edge(self, e):
        self.edges.append(e)

    def get_posterior(self, theta):
        # Link edges to dataset
        pass