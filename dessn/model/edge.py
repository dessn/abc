import abc


class Edge(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, probability_of, given):
        if type(probability_of) != list:
            probability_of = [probability_of]
        if type(given) != list:
            given = [given]
        for name in probability_of:
            assert type(name) == str, "Parameter name should be a string, but got %s" % name
        for name in given:
            assert type(name) == str, "Parameter name should be a string, but got %s" % name
        self.probability_of = probability_of
        self.given = given

    @abc.abstractmethod
    def get_log_likelihood(self, data):
        return

