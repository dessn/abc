import abc


class Node(object):
    """ A node represented on a PGM model. Normally encapsulated by a single parameter, or several related parameters.

    The Node class can essentially be thought of as a wrapper around a parameter or variable in your model. However,
    as some parameters are highly related (for example, flux and flux error), Nodes allow you to declare multiple
    parameters.

    This class is an abstract class, and cannot be directly instantiated. Instead, instantiate one of the provided
    subclasses, as detailed below.

    Parameters
    ----------
    node_name : str
        The node name, only used when plotting on a PGM
    names : str or list[str]
        The model parameter encapsulated by the node, or list of model parameters
    labels : str or list[str]
        Latex ready labels for the given names. Used in the PGM and corner plots.
    parameter_type : :class:`.NodeType`
        The type of subclass. Informs the model how to utilise the node.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, label, group=None):
        assert type(name) == str, "The name of this node, %s, is not a string" % name
        assert type(label) == str, "Supplied name %s is not a string or list" % label
        self.name = name
        self.label = label
        self.group = group

    @abc.abstractmethod
    def get_suggestion_requirements(self):
        """ Returns suggestion parameter requirements.
        """
        return

    @abc.abstractmethod
    def get_suggestion(self, data):
        return


class NodeObserved(Node):
    """ A node representing one or more observed variables

    This node is used for all observables in the model. In addition to a normal node, it also contains data,
    which can be in arbitrary format. This data is what is given to the incoming and outgoing node edges
    to calculate likelihoods.

    Parameters
    ----------
    node_name : str
        The node name, only used when plotting on a PGM
    names : str or list[str]
        The model parameter encapsulated by the node, or list of model parameters
    labels : str or list[str]
        Latex ready labels for the given names. Used in the PGM and corner plots.
    datas : object or list[obj]
        One data object for each supplied parameter name. **Must** be the same length as names if names is a list.
    """
    def __init__(self, name, label, data, group=None):
        assert type(data) == list, "Data must be a list"
        self.data = data
        super(NodeObserved, self).__init__(name, label, group=group)

    def get_data(self):
        """ Returns a dictionary containing keys of the parameter names and values of the parameter data object
        """
        return {self.name: self.data}

    def get_suggestion(self, data):
        return []

    def get_suggestion_requirements(self):
        return []


class NodeUnderlying(Node):
    r""" A node representing an underlying parameter in your model.

    On the PGM, these nodes would be at the very top, and would represent the variables
    we are trying to fit for, such as :math:`\Omega_M`.

    These nodes are required to implement the abstract method ``get_log_prior``

    Parameters
    ----------
    node_name : str
        The node name, only used when plotting on a PGM
    names : str or list[str]
        The model parameter encapsulated by the node, or list of model parameters
    labels : str or list[str]
        Latex ready labels for the given names. Used in the PGM and corner plots.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, names, labels, group=None):
        super(NodeUnderlying, self).__init__(names, labels, group=group)

    def get_log_prior(self, data):
        """ Returns the log prior for the parameter.

        Parameters
        ----------
        data : dic
            A dictionary containing all data and the model parameters being tested at a given step
            in the MCMC chain. For this class, if the class was instantiated with a name of "omega_m",
            the input dictionary would have the key "omega_m", and the value of "omega_m" at that
            particular step in your chain.

        Returns
        -------
        float
            the log prior probability given the current value of the parameters
        """
        return 1

    def get_suggestion_requirements(self):
        return []


class NodeTransformation(Node):
    """ A node representing a variable transformation.

    This node essentially represents latent variables which are fully determined - their probability is given by
    a delta function. Examples of this might be the luminosity distance, as it is known exactly when given cosmology
    and redshift. Or it might represent a conversion between observed flux and actual flux, given we have a well
    defined flux correction.

    On a PGM, this node would be represented by a point, not an ellipse.

    Note that this node declares all associated parameters to be transformation parameters,
    although the transformation functions themselves are defined by the edges into and out of this node.

    Parameters
    ----------
    node_name : str
        The node name, only used when plotting on a PGM
    names : str or list[str]
        The model parameter encapsulated by the node, or list of model parameters
    labels : str or list[str]
        Latex ready labels for the given names. Used in the PGM and corner plots.
    """
    def __init__(self, names, labels, group=None):
        super(NodeTransformation, self).__init__(names, labels, group=group)

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return []


class NodeLatent(Node):
    """ A node representing a latent, or hidden, variable in our model.

    Given infinitely powerful computers, these nodes would not be necessary, for they represent
    marginalisation over unknown / kidden / latent parameters in the model, and we would simple integrate them
    out when computing the likelihood probability. However, this is not the case, and it is more efficient to
    simply incorporate latent parameters into our model and essentially marginalise over them using Monte Carlo
    integration. We thus trade explicit numerical integration in each step of our calculation for increased dimensionality.

    For examples on why and how to use latent parameters, see the examples beginning in :class:`.Example`.

    Parameters
    ----------
    node_name : str
        The node name, only used when plotting on a PGM
    names : str or list[str]
        The model parameter encapsulated by the node, or list of model parameters
    labels : str or list[str]
        Latex ready labels for the given names. Used in the PGM and corner plots.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, names, labels, group=None):
        super(NodeLatent, self).__init__(names, labels, group=group)

    @abc.abstractmethod
    def get_num_latent(self):
        """ The number of latent parameters to include in the model.

        Running MCMC requires knowing the dimensionality of our model, which means knowing how many
        latent parameters (realisations of an underlying hidden distribution) we require.

        For example, if we observe a hundred supernova drawn from an underlying supernova distribution,
        we would have to realise a hundred latent variables - one per data point.

        Returns
        -------
        int
            the number of latent parameters required by this node
        """
        pass


class NodeDiscrete(Node):

    __metaclass__ = abc.ABCMeta

    def __init__(self, name, label, group=None):
        # TODO Enforce name and label are not lists for a discrete node
        super(NodeDiscrete, self).__init__(name, label, group=group)

    @abc.abstractmethod
    def get_discrete(self, data):
        pass

    @abc.abstractmethod
    def get_discrete_requirements(self):
        pass

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return []