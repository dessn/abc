from dessn.model.node import Node, NodeObserved, NodeLatent, NodeUnderlying, NodeTransformation
from dessn.model.edge import EdgeTransformation
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
        elif isinstance(node, NodeLatent):
            self._latent_nodes.append(node)
        elif isinstance(node, NodeTransformation):
            self._transformation_nodes.append(node)
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
                    assert len(self._in[name]) > 0, "Internal parameter %s has no incoming edges" % name
                    assert len(self._out[name]) > 0, "Internal parameter %s does not have any outgoing edges" % name
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
        self._validate_model()
        self._create_data_structures()
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

    def _get_log_prior(self, theta_dict):
        result = []
        for node in self._underlying_nodes:
            result.append(node.get_log_prior({key: theta_dict[key] for key in node.names}))
        return np.sum(result)

    def _get_transformation(self, theta_dict, edge):
        return edge.get_transformation({key: theta_dict[key] for key in edge.probability_of})

    def _get_log_likelihood(self, theta_dict, edge):
        return edge.get_log_likelihood({key: theta_dict[key] for key in edge.given + edge.probability_of})

    def fit_model(self, num_walkers=None, num_steps=30000, num_burn=27000, filename=None):
        # TODO: Refactor this section, it should not be encoded in the model
        num_dim = len(self._theta_names)
        self.logger.debug("Fitting model with %d dimensions" % num_dim)
        if num_walkers is None:
            num_walkers = num_dim * 16
        start = np.zeros((num_walkers, num_dim))
        self.logger.debug("Generating starting guesses")
        for row in range(num_walkers):
            for i, name in enumerate(self._theta_names):
                if name == "SN_theta_1":
                    start[row, i] = np.random.normal(1,0.2) * 100
                    # print(start[row,i])
                elif name == "SN_theta_2":
                    start[row, i] = np.random.normal(1,0.2) * 20
                elif name =="luminosity":
                    start[row, i] = np.random.normal(1,0.3) * 100

                # TODO: Actual start guesses
        self.logger.debug("Running emcee")
        sampler = emcee.EnsembleSampler(num_walkers, num_dim, self._get_log_posterior, live_dangerously=True)
        for i, result in enumerate(sampler.sample(start, iterations=num_steps)):
            if i % 100 == 0:
                print(i)
        self.logger.debug("Getting emcee chain")
        sample = sampler.chain[:, num_burn:, :self._num_actual]  # discard burn-in points
        sample = sample.reshape((-1, self._num_actual))
        self.sampler = sampler
        self.sample = sample
        means = np.mean(sampler.chain[:, num_burn:, :].reshape((-1, num_dim)), axis=0)
        print (means)
        diffs = means[2:] - self.flux.datas[0]
        print(diffs)

        self.logger.debug("Creating corner plot")
        fig = corner.corner(sample, labels=self._theta_labels[:self._num_actual])
        if filename is not None:
            filename = os.path.dirname(__file__) + os.sep + ("../../plots/%s" % filename)
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        plt.show()

        self.logger.debug("Getting starting locations")
