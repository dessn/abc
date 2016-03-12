import abc


class Edge(object):
    """ An edge connection one or more parameters to one or more different parameters.

    An edge is a connection between parameters (*not* Nodes), and signifies a joint probability distribution.
    That is, if in our mathematical definition of our model, we find the term :math:`P(a,b|c,d,e)`, this
    would be represented by a single edge. Similarly, :math:`P(a|b)P(b|c,d)` would be two edges.

    Parameters
    ----------
    probability_of : str or list[str]
        The dependent parameters. With the example :math:`P(a,b|c,d)`, this input would be ``['a','b']``.
    given : str or list[str]
        In the example :math:`P(a,b|c,d)`, this input would be ``['c','d']``.
    """
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
        r""" Gets the log likelihood of this edge.

        For example, if we had

        .. math::
            P(a,b|c,d) = \frac{1}{\sqrt{2\pi}d} \exp\left(-\frac{(ab-c)^2}{d^2}\right),

        we could implement this function as ``return -np.log(np.sqrt(2*np.pi)*data['d']) - (data['a']*data['b'] - data['c'])**2/(data['d']**2)``

        Returns
        -------
        float
            the log likelihood given the supplied data and the model parametrisation.
        """
        return


class EdgeTransformation(Edge):
    """ This specialised edge is used to connect to transformation nodes.

    A transformation edge does not give a likelihood, but - as it is a known transformation - returns a dictionary
    when `get_transformation` is invoked that is injected into the data dictionary given to regular edges.

    See :class:`.LuminosityToAdjusted` for a simple example.

    Parameters
    ----------
    probability_of : str or list[str]
        The dependent parameters. With the example :math:`P(a,b|c,d)`, (assuming the functional form is a delta), this input would be ``['a','b']``.
    given : str or list[str]
        In the example :math:`P(a,b|c,d)`, (assuming the functional form is a delta), this input would be ``['c','d']``.
    """
    def __init__(self, probability_of, given):
        super(EdgeTransformation, self).__init__(probability_of, given)

    @abc.abstractmethod
    def get_transformation(self, data):
        """ Calculates the new parameters from the given data

        Returns
        -------
        dict
            a dictionary containing a value for each parameter given in ``transform_to``
        """
        pass

    def get_log_likelihood(self, data):
        return 0
