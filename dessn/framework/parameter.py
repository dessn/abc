import abc
import numpy as np

class Parameter(object):
    """ A parameter represented on a node on a PGM framework. Multiple parameters can be assigned to the same node on a
    PGM by giving them the same group.

    The Parameter class can essentially be thought of as a wrapper around a parameter or variable in your framework.

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
        """ This function is used when finding a good starting position for the sampler.

        The better this suggestion is, the less burn in time is needed. As parameters can
        vary by orders of magnitude, and have allowed ranges (some parameters cannot be negative
        for example), local optimisation methods do not always work when starting with some arbitrary
        and random initial condition. As such, overriding this parameter is required for all
        latent and underlying parameters.

        Parameters
        ----------
        data : dictionary
            The parameters and observed data to generate a suggested parameter from

        Returns
        -------
            float
                a suggested parameter
        """
        return

    @abc.abstractmethod
    def get_suggestion_sigma(self, data):
        """ Starting all walkers from the same position is not a good thing to do, so the
        suggested starting positions given by the :func:`get_suggestion` need to be randomised
        slightly so that the walkers start in different positions. This is done by taking the suggested
        parameter and adding uniform noise (**not** gaussian any more) to it,
        where the upper or lower maximum deviation of the suggested parameter
        is given by this function. Overestimating this value to try and ensure a proper spread of
        walker positions can lead to complications and increased convergence, so don't always think
        bigger is better!

        Parameters
        ----------
        data : dictionary
            The parameters and observed data to generate a suggested parameter from

        Returns
        -------
            float
                a suggested parameter sigma, used to randomise the suggest parameter

        """
        return


class ParameterObserved(Parameter):
    """ A parameter representing an observed variables

    This parameter is used for all observables in the framework. In addition to a normal parameter, it also contains data,
    which should be a list of ``n`` elements (for ``n`` data points), with each element allowed to be an
    arbitrary data type. This data is what is given to the incoming and outgoing node edges
    to calculate likelihoods. It is **very important** that, if your framework has multiple observed parameters, each
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
        assert type(data) in [list, np.ndarray], \
            "Data must be a list format"
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
    r""" A parameter representing an underlying parameter in your framework.

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
            A dictionary containing all data and the framework parameters being tested at a given step
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
    """ A parameter representing a latent, or hidden, variable in our framework.

    Given infinitely powerful computers, these nodes would not be necessary, for they represent
    marginalisation over unknown / hidden / latent parameters in the framework, and we would simple integrate them
    out when computing the likelihood probability. However, this is not the case, and it is more efficient to
    simply incorporate latent parameters into our framework and essentially marginalise over them using Monte Carlo
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
        """ The number of latent parameters to include in the framework.

        Running MCMC requires knowing the dimensionality of our framework, which means knowing how many
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
    """ A parameter representing a discrete variable in our framework.

    Unlike latent variables which can be easy marginalised over, discrete variables simply create more issues than
    they are worth. Algorithms like Hamiltonian Monte Carlo require continuous posterior surfaces, which - off the bat -
    simply rule out discrete parameters. As such, discrete parameters in the framework are integrated out (really, they
    are summed over). Examples of discrete parameters might be supernova types, which galaxy is the actual transient
    host, or trivially whether a coin was flipped to be heads or tails.

    Discrete parameters must implement a :func:`.get_discrete` method, which needs to return the discrete options
    for the particular step in the framework. Some discrete options may be global (for example, we would consider all
    supernova type combinations with each supernova), however some can be dependent on the current observation
    (some supernova might have only one possible host, others might have two or more). As the possible types
    can be dependent on observation, the :func:`.get_discrete_requirements` method also exists and can be
    overridden. It functions identically to the ``get_suggestion_requirements`` method in the :class:`ParameterLatent`
    class.

    For examples how to use latent parameters, see the example give by :class:`.DiscreteModel`.

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
        super(ParameterDiscrete, self).__init__(name, label, group=group)

    @abc.abstractmethod
    def get_discrete(self, data):
        """ Returns the possible discrete types for this parameter.

        Parameters
        ----------
        data : dictionary
            Contains parameter values and observations for the particular step in the chain and observation

        Returns
        -------
            list
                A list of types which are iterated over.
        """
        pass

    def get_discrete_requirements(self):
        """ Gets the data and parameters required for generating the discrete values for this parameters

        Returns
        -------
            list
                Defaults to an empty list.
        """
        return []

    def get_suggestion_requirements(self):
        return []

    def get_suggestion(self, data):
        return

    def get_suggestion_sigma(self, data):
        return
