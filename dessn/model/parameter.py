import abc


class Parameter(object):
    """ A parameter represented on a node on a PGM model. Multiple parameters can be assigned to the same node on a
    PGM by giving them the same group.

    The Parameter class can essentially be thought of as a wrapper around a parameter or variable in your model.

    This class is an abstract class, and cannot be directly instantiated. Instead, instantiate one of the provided
    subclasses, as detailed below.

    Parameters
    ----------
    name : str
        The parameter name, used as the key to access this parameter in the data object
    label : str
        The parameter label, for use in plotting and PGM creation.
    group : str, optional
        The group in the PGM that this parameter belongs to. Will replace ``name`` on the PGM if set.

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

    @abc.abstractmethod
    def get_suggestion_sigma(self, data):
        return


class ParameterObserved(Parameter):
    """ A parameter representing an observed variables

    This parameter is used for all observables in the model. In addition to a normal parameter, it also contains data,
    which should be a list of ``n`` elements (for ``n`` data points), with each element allowed to be an
    arbitrary data type. This data is what is given to the incoming and outgoing node edges
    to calculate likelihoods. It is **very important** that, if your model has multiple observed parameters, each
    observed parameter returns lists of the same length.

    Parameters
    ----------
    name : str
        The parameter name, used as the key to access this parameter in the data object
    label : str
        The parameter label, for use in plotting and PGM creation.
    data : list[object]
        The data list to supply to the edges.
    group : str, optional
        The group in the PGM that this parameter belongs to. Will replace ``name`` on the PGM if set.
    """
    def __init__(self, name, label, data, group=None):
        assert type(data) == list, "Data must be a list. If you are passing in a :class:`np.ndarray`, call ``.tolist()``"
        self.data = data
        super(ParameterObserved, self).__init__(name, label, group=group)

    def get_data(self):
        """ Returns a dictionary containing keys of the parameter names and values of the parameter data object
        """
        return {self.name: self.data}

    def get_suggestion(self, data):
        return

    def get_suggestion_requirements(self):
        return []

    def get_suggestion_sigma(self, data):
        return


class ParameterUnderlying(Parameter):
    r""" A parameter representing an underlying parameter in your model.

    On the PGM, these nodes would be at the very top, and would represent the variables
    we are trying to fit for, such as :math:`\Omega_M`.

    These nodes are required to implement the abstract method ``get_log_prior``. In addition,
    they must also implement the abstract method ``get_suggestion``, which returns a suggested
    value for the parameter, given some input specified by the ``get_suggestion_requirements``.
    By default ``get_suggestion_requirements`` specifies no requirements, however you can override
    this method, such that it returns a list of required data to generate a suggestion.

    Parameters
    ----------
    name : str
        The parameter name, used as the key to access this parameter in the data object
    label : str
        The parameter label, for use in plotting and PGM creation.
    group : str, optional
        The group in the PGM that this parameter belongs to. Will replace ``name`` on the PGM if set.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, label, group=None):
        super(ParameterUnderlying, self).__init__(name, label, group=group)

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


class ParameterTransformation(Parameter):
    """ A parameter representing a variable transformation.

    This parameter essentially represents latent variables which are fully determined - their probability is given by
    a delta function. Examples of this might be the luminosity distance, as it is known exactly when given cosmology
    and redshift. Or it might represent a conversion between observed flux and actual flux, given we have a well
    defined flux correction.

    On a PGM, this parameter would be represented by a point, not an ellipse.

    Parameters
    ----------
    name : str
        The parameter name, used as the key to access this parameter in the data object
    label : str
        The parameter label, for use in plotting and PGM creation.
    group : str, optional
        The group in the PGM that this parameter belongs to. Will replace ``name`` on the PGM if set.
    """
    def __init__(self, name, label, group=None):
        super(ParameterTransformation, self).__init__(name, label, group=group)

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return

    def get_suggestion_sigma(self, data):
        return


class ParameterLatent(Parameter):
    """ A parameter representing a latent, or hidden, variable in our model.

    Given infinitely powerful computers, these nodes would not be necessary, for they represent
    marginalisation over unknown / hidden / latent parameters in the model, and we would simple integrate them
    out when computing the likelihood probability. However, this is not the case, and it is more efficient to
    simply incorporate latent parameters into our model and essentially marginalise over them using Monte Carlo
    integration. We thus trade explicit numerical integration in each step of our calculation for increased dimensionality.

    For examples on why and how to use latent parameters, see the examples beginning in :class:`.Example`.

    Parameters
    ----------
    name : str
        The parameter name, used as the key to access this parameter in the data object
    label : str
        The parameter label, for use in plotting and PGM creation.
    group : str, optional
        The group in the PGM that this parameter belongs to. Will replace ``name`` on the PGM if set.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, label, group=None):
        super(ParameterLatent, self).__init__(name, label, group=group)

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


class ParameterDiscrete(Parameter):

    __metaclass__ = abc.ABCMeta

    def __init__(self, name, label, group=None):
        super(ParameterDiscrete, self).__init__(name, label, group=group)

    @abc.abstractmethod
    def get_discrete(self, data):
        pass

    @abc.abstractmethod
    def get_discrete_requirements(self):
        pass

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return

    def get_suggestion_sigma(self, data):
        return
