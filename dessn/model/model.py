from dessn.model.node import Node, NodeObserved, NodeLatent, NodeUnderlying, NodeTransformation
import numpy as np
import logging
import emcee
import corner
import matplotlib.pyplot as plt
import os


class Model(object):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nodes = []
        self.edges = []
        self._node_dict = {}
        self._node_indexes = {}
        self._observed_nodes = []
        self._internal_nodes = []
        self._underlying_nodes = []
        self._in = {}
        self._out = {}
        self._theta_names = []
        self._num_actual = None
        self.data = {}
        self._data_edges = {}

    def add_node(self, node):
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
            elif isinstance(node, NodeTransformation) or isinstance(node, NodeLatent):
                self._internal_nodes.append(node)
            elif isinstance(node, NodeUnderlying):
                self._underlying_nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)
        for p in edge.probability_of:
            for g in edge.given:
                self._in[g].append(p)
                self._out[p].append(g)

    def _validate_model(self):
        assert len(self._underlying_nodes) > 0, "No underlying model to constrain"
        assert len(self._observed_nodes) > 0, "No observed nodes found"
        for node in self.nodes:
            for name in node.names:
                if isinstance(node, NodeObserved):
                    assert len(self._in[name]) == 0, "Observed parameter %s should not have incoming edges" % name
                    assert len(self._out[name]) > 0, "Observed parameter %s is not utilised in the PGM" % name
                elif isinstance(node, NodeLatent) or isinstance(node, NodeTransformation):
                    assert len(self._in[name]) > 0, "Latent parameter %s has no incoming edges" % name
                    assert len(self._out[name]) > 0, "Latent parameter %s does not have any outgoing edges" % name
                elif isinstance(node, NodeUnderlying):
                    assert len(self._in[name]) > 0, "Underlying parameter %s has no incoming edges" % name
                    assert len(self._out[name]) == 0, "Underlying parameter %s should not have an outgoing edge" % name

    def _create_data_structures(self):
        for node in self._observed_nodes:
            self.data.update(node.get_data())
        for edge in self.edges:
            self._data_edges[edge] = [self.data[g] for g in edge.given if g in self.data.keys()]
        for node in self._underlying_nodes:
            for name in node.names:
                self._theta_names.append(name)
        self._num_actual = len(self._theta_names)
        for node in self._internal_nodes:
            for name in node.names:
                self._theta_names += [name] * node.get_num_latent()

    def finalise(self):
        self._validate_model()
        self._create_data_structures()
        print(self._theta_names)
        print(self.data)
        self.logger.info("Model validation passed")

    def _get_theta_dict(self, theta):
        result = self.data.copy()
        for theta, theta_name in zip(theta, self._theta_names):
            if self._theta_names.count(theta_name) == 1:
                result[theta_name] = theta
            else:
                if theta_name not in result:
                    result[theta_name] = []
                result[theta_name].append(theta)
        return result

    def _get_log_posterior(self, theta):

        theta_dict = self._get_theta_dict(theta)
        probability = self._get_log_prior(theta_dict)
        probability += self._get_log_likelihood(theta_dict)
        return probability

    def _get_log_prior(self, theta_dict):
        result = []
        for node in self._underlying_nodes:
            result.append(node.get_log_prior({key: theta_dict[key] for key in node.names}))
        return np.sum(result)

    def _get_log_likelihood(self, theta_dict):
        result = []
        for edge in self.edges:
            # TODO: Change it so that node transformations are called first and in order
            # and invoke get_transformation and inject the results into data
            result.append(edge.get_log_likelihood({key: theta_dict[key] for key in edge.given + edge.probability_of}))
        return np.sum(result)

    def fit_model(self, num_walkers=None, num_steps=1000, num_burn=500, filename=None):
        # TODO: Refactor this section, it should not be encoded in the model
        num_dim = len(self._theta_names)
        if num_walkers is None:
            num_walkers = 2 * num_dim
        start = np.zeros((num_walkers, num_dim))
        self.logger.debug("Generating starting guesses")
        for row in range(num_walkers):
            for i, name in enumerate(self._theta_names):
                start[row, i] = np.random.normal(100,30)
                # TODO: Actual start guesses
        self.logger.debug("Running emcee")
        sampler = emcee.EnsembleSampler(num_walkers, num_dim, self._get_log_posterior, live_dangerously=True)
        for i, result in enumerate(sampler.sample(start, iterations=num_steps)):
            print(i)
        self.logger.debug("Getting emcee chain")
        sample = sampler.chain[:, num_burn:, :self._num_actual]  # discard burn-in points
        sample = sample.reshape((-1, self._num_actual))
        self.sampler = sampler
        self.sample = sample
        self.logger.debug("Creating corner plot")
        fig = corner.corner(sample)
        plt.show()
        if filename is not None:
            filename = os.path.dirname(__file__) + os.sep + ("../../plots/%s" % filename)
        fig.savefig(filename, bbox_inches='tight', dpi=300)

        self.logger.debug("Getting starting locations")
