from dessn.model.node import Node, NodeObserved, NodeLatent, NodeUnderlying, NodeTransformation
from dessn.model.edge import EdgeTransformation
from dessn.utility.newtonian import NewtonianPosition
from dessn.utility.hdemcee import EmceeWrapper
from dessn.chain.chain import ChainConsumer
import numpy as np
import logging
import emcee
from emcee.utils import MPIPool
import corner
import matplotlib.pyplot as plt
from matplotlib import rc
import daft
import sys
from scipy.optimize import fmin_bfgs


class Model(object):
    """ A generalised model for use in arbitrary situations.

    A model is, at heart, simply a collection of nodes and edges. Apart from simply
    being a container in which to place nodes and edges, the model is also responsible for
    figuring out how to connect edges (which map to parameters) with the right nodes, for sorting edges
    such that when an edge is evaluated all its required data has been generated by other nodes or edges,
    for managing the ``emcee`` running, and also for generating the visual PGMs.

    It is thus a complex class, and I expect, as of writing this summary, it contains numerous bugs.

    Parameters
    ----------
    model_name : str
        The model name, used for serialisation
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.nodes = []
        self.edges = []
        self._node_dict = {}
        self._node_indexes = {}
        self._observed_nodes = []
        self._latent_nodes = []
        self._transformation_nodes = []
        self._underlying_nodes = []
        self._in = {}
        self._out = {}
        self._theta_names = []
        self._theta_labels = []
        self._ordered_edges = []
        self._num_actual = None
        self.data = {}
        self._data_edges = {}
        self._finalised = False
        self.flat_chain = None

    def add_node(self, node):
        """ Adds a node into the models collection of nodes.

        Parameter
        ---------
        node : :class:`.Node`
        """
        assert isinstance(node, Node), "Supplied parameter is not a recognised Node object"
        for name in node.names:
            assert name not in self._node_dict.keys(), "Parameter %s is already in the model" % name
        self.nodes.append(node)
        for name in node.names:
            self._node_dict[name] = node
            self._in[name] = []
            self._out[name] = []
        if isinstance(node, NodeObserved):
            self._observed_nodes.append(node)
        elif isinstance(node, NodeLatent):
            self._latent_nodes.append(node)
        elif isinstance(node, NodeTransformation):
            self._transformation_nodes.append(node)
        elif isinstance(node, NodeUnderlying):
            self._underlying_nodes.append(node)
        self._finalised = False

    def add_edge(self, edge):
        """ Adds an edge into the models collection of edges

        Parameter
        ---------
        edge : :class:`.Edge`
        """
        self.edges.append(edge)
        for p in edge.probability_of:
            for g in edge.given:
                self._in[g].append(p)
                self._out[p].append(g)
        self._finalised = False

    def _validate_model(self):
        assert len(self._underlying_nodes) > 0, "No underlying model to constrain"
        assert len(self._observed_nodes) > 0, "No observed nodes found"
        for node in self.nodes:
            for name in node.names:
                if isinstance(node, NodeObserved):
                    assert len(self._in[name]) == 0, "Observed parameter %s should not have incoming edges" % name
                    assert len(self._out[name]) > 0, "Observed parameter %s is not utilised in the PGM" % name
                elif isinstance(node, NodeLatent) or isinstance(node, NodeTransformation):
                    assert len(self._in[name]) > 0, "Internal parameter %s has no incoming edges" % name
                    # assert len(self._out[name]) > 0, "Internal parameter %s does not have any outgoing edges" % name
                elif isinstance(node, NodeUnderlying):
                    assert len(self._in[name]) > 0, "Underlying parameter %s has no incoming edges" % name
                    assert len(self._out[name]) == 0, "Underlying parameter %s should not have an outgoing edge" % name

    def _create_data_structures(self):
        for node in self._observed_nodes:
            self.data.update(node.get_data())
        for edge in self.edges:
            self._data_edges[edge] = [self.data[g] for g in edge.given if g in self.data.keys()]
        for node in self._underlying_nodes:
            for name, label in zip(node.names, node.labels):
                self._theta_names.append(name)
                self._theta_labels.append(label)
        self._num_actual = len(self._theta_names)
        for node in self._latent_nodes:
            for name in node.names:
                self._theta_names += [name] * node.get_num_latent()

        num_edges = len(self.edges)
        observed_names = [key for key in self._in if len(self._in[key]) == 0]
        self._ordered_edges = []
        sort_req = {}
        count = 0
        max_count = 1000
        while len(self._ordered_edges) < num_edges:
            for edge in self.edges:
                unsatisfied_requirements = [r for r in edge.probability_of if r not in observed_names]
                if len(unsatisfied_requirements) == 0:
                    self._ordered_edges.append(edge)
                    for p in edge.probability_of:
                        for g in edge.given:
                            if g not in sort_req:
                                sort_req[g] = []
                            sort_req[g].append(p)
                    for g in edge.given:
                        if set(self._in[g]) == set(sort_req[g]):
                            observed_names.append(g)
            count += 1
            if count > max_count:
                raise ValueError("Model edges cannot be ordered. Please double check your edges")

    def finalise(self):
        """ Finalises the model.

        This method runs consistency checks on the model (making sure there are not orphaned
        nodes, edges to parameters that do not exist, etc), and in doing so links the right
        edges to the right nodes and determines the order in which edges should be evaluated.

        You can manually call this method after setting all nodes and edges to confirm as early
        as possible that the model is valid. If you do not call it manually, this method
        is invoked by the model when requesting concrete information, such as the PGM or model fits.
        """
        self._validate_model()
        self._create_data_structures()
        self._finalised = True
        self.logger.info("Model validation passed")

    def _get_theta_dict(self, theta):
        result = self.data.copy()
        make_array = []
        for theta, theta_name in zip(theta, self._theta_names):
            if self._theta_names.count(theta_name) == 1:
                result[theta_name] = theta
            else:
                if theta_name not in result:
                    result[theta_name] = []
                    make_array.append(theta_name)
                result[theta_name].append(theta)
        for m in make_array:
            result[m] = np.array(result[m])
        return result

    def _get_log_posterior(self, theta):

        theta_dict = self._get_theta_dict(theta)
        probability = self._get_log_prior(theta_dict)
        for edge in self._ordered_edges:
            if isinstance(edge, EdgeTransformation):
                theta_dict.update(self._get_transformation(theta_dict, edge))
            else:
                probability += self._get_log_likelihood(theta_dict, edge)
        return probability

    def _get_negative_log_posterior(self, theta):
        return -self._get_log_posterior(theta)

    def _get_starting_position(self, num_walkers):
        num_dim = len(self._theta_names)
        self.logger.debug("Generating starting guesses")
        p0 = np.ones(num_dim)
        optimised = fmin_bfgs(self._get_negative_log_posterior, p0, disp=False)
        self.logger.debug("Starting position is: %s" % optimised)

        std = np.random.uniform(0.5, 1.5, size=(num_walkers, num_dim))
        start = std * optimised
        return start

    def _get_log_prior(self, theta_dict):
        result = []
        for node in self._underlying_nodes:
            result.append(node.get_log_prior({key: theta_dict[key] for key in node.names}))
        return np.sum(result)

    def _get_transformation(self, theta_dict, edge):
        return edge.get_transformation({key: theta_dict[key] for key in edge.probability_of})

    def _get_log_likelihood(self, theta_dict, edge):
        return edge.get_log_likelihood({key: theta_dict[key] for key in edge.given + edge.probability_of})

    def get_pgm(self, filename=None):
        """ Renders (and returns) a PGM of the current model.

        Parameters
        ----------
        filename : str, optional
            if the filename is set, the PGM is saved to file in the top level ``plots`` directory.

        Returns
        -------
        :class:`daft.PGM`
            The ``daft`` PGM class, for further customisation if required.
        """
        if not self._finalised:
            self.finalise()

        self.logger.info("Generating PGM")
        rc("font", family="serif", size=8)
        rc("text", usetex=True)

        x_size = 9
        y_size = 8
        border = 1
        node_name_dict = {}

        n = []
        e = []
        t = []
        b = []

        for node in self.nodes:
            n.append(node.node_name)
            if node in self._observed_nodes:
                b.append(node.node_name)
            if node in self._underlying_nodes:
                t.append(node.node_name)
            for name in node.names:
                node_name_dict[name] = node.node_name

        for edge in self.edges:
            for g in edge.given:
                for p in edge.probability_of:
                    e.append([node_name_dict[g], node_name_dict[p]])

        self.logger.debug("Using Newtonian positioner to position %d nodes and %d edges" % (len(n), len(e)))

        positioner = NewtonianPosition(n, e, top=t, bottom=b)
        x, y = positioner.fit()
        x = (x_size - 2 * border) * x + border
        y = (y_size - 2 * border) * y + border

        self.logger.debug("Creating PGM from positioner results")
        pgm = daft.PGM([x_size, y_size], origin=[0., 0.2], observed_style='inner')
        for node, x, y in zip(self.nodes, x, y):
            obs = node in self._observed_nodes
            fixed = node in self._transformation_nodes
            node_name = node.node_name
            node_label = node_name.replace(" ", "\n") + "\n" + ", ".join(node.labels)

            pgm.add_node(daft.Node(node_name, node_label, x, y, scale=1.6, aspect=1.3, observed=obs, fixed=fixed))

        for edge in self.edges:
            for g in edge.given:
                for p in edge.probability_of:
                    pgm.add_edge(node_name_dict[g], node_name_dict[p])
        pgm.render()
        if filename is not None:
            self.logger.debug("Saving figure to %s" % filename)
            pgm.figure.savefig(filename, transparent=True, dpi=300)

        return pgm

    def fit_model(self, num_walkers=None, num_steps=5000, num_burn=3000, temp_dir=None, save_interval=300):
        """ Uses ``emcee`` to fit the supplied model.

        This method sets an emcee run using the ``EnsembleSampler`` and manual chain management to allow for
        very high dimension models. MPI running is detected automatically for less hassle, and chain progress is
        serialised to disk automatically for convenience.

        This method works... but is still a work in progress

        Parameters
        ----------
        num_walkers : int, optional
            The number of walkers to run. If not supplied, it defaults to eight times the model dimensionality
        num_steps : int, optional
            The number of steps to run
        num_burn : int, optional
            The number of steps to discard for burn in
        temp_dir : str
            If set, specifies a directory in which to save temporary results, like the emcee chain
        save_interval : float
            The amount of seconds between saving the chain to file. Setting to ``None`` disables serialisation.

        Returns
        -------
        ndarray
            The final flattened chain of dimensions ``(num_dimensions, num_walkers * (num_steps - num_burn))``
        fig
            The corner plot figure returned from ``corner.corner(...)``
        """
        if not self._finalised:
            self.finalise()
        pool = None
        try:
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
        except ImportError:
            self.logger.info("mpi4py is not installed or not configured properly. Ignore if running through python, not mpirun")
        except ValueError:
            self.logger.info("Unable to start MPI pool, expected normal python execution")

        num_dim = len(self._theta_names)
        self.logger.debug("Fitting model with %d dimensions" % num_dim)
        if num_walkers is None:
            num_walkers = num_dim * 8

        self.logger.debug("Running emcee")
        sampler = emcee.EnsembleSampler(num_walkers, num_dim, self._get_log_posterior, pool=pool)
        emcee_wrapper = EmceeWrapper(sampler)
        flat_chain = emcee_wrapper.run_chain(num_steps, num_burn, num_walkers, num_dim, start=self._get_starting_position, save_dim=self._num_actual, temp_dir=temp_dir, save_interval=save_interval)
        self.logger.debug("Fit finished")
        self.flat_chain = flat_chain
        return flat_chain, self._theta_names[:self._num_actual], self._theta_labels[:self._num_actual]

    def corner(self, filename=None):
        assert self.flat_chain is not None, "You have to run fit_model before calling corner"
        self.logger.debug("Creating corner plot")
        fig = corner.corner(self.flat_chain, labels=self._theta_labels[:self._num_actual], quantiles=[0.16, 0.5, 0.84], bins=100)
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
        plt.show()

    def chain_plot(self, **kwargs):
        chain_plotter = ChainConsumer(self.flat_chain, self._theta_labels[:self._num_actual])
        chain_plotter.plot(**kwargs)

    def chain_summary(self):
        chain_plotter = ChainConsumer(self.flat_chain, self._theta_labels[:self._num_actual])
        print(chain_plotter.get_summary())