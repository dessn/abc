from ...framework.model import Model
from ...framework.parameter import ParameterUnderlying, ParameterObserved, \
    ParameterDiscrete, ParameterLatent, ParameterTransformation
from ...framework.edge import Edge, EdgeTransformation
import numpy as np


class ObservedValue(ParameterObserved):
    def __init__(self, data):
        super().__init__("obs", "$o$", data)


class LatentValue1(ParameterLatent):
    def __init__(self, n):
        super().__init__("latent1", "latent1")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion(self, data):
        return data["obs"]

    def get_suggestion_sigma(self, data):
        return np.ones(data["obs"].shape)

    def get_suggestion_requirements(self):
        return ["obs"]


class Transformed1(ParameterTransformation):
    def __init__(self):
        super().__init__("t1", "t1")


class DiscreteValue1(ParameterDiscrete):
    def __init__(self):
        super().__init__("discrete", "$l$")

    def get_discrete(self, data):
        return [["r", "b"] for a in data["obs"]]

    def get_discrete_requirements(self):
        return ["obs"]


class DiscreteValue2(ParameterDiscrete):
    def __init__(self):
        super().__init__("discrete", "$l$")

    def get_discrete(self, data):
        return "r", "b"

    def get_discrete_requirements(self):
        return []


class Transformed2(ParameterTransformation):
    def __init__(self):
        super().__init__("t2", "t2")


class LatentValue2(ParameterLatent):
    def __init__(self, n):
        super().__init__("latent2", "latent2")
        self.n = n

    def get_num_latent(self):
        return self.n

    def get_suggestion(self, data):
        return data["obs"]

    def get_suggestion_sigma(self, data):
        return np.ones(data["obs"].shape)

    def get_suggestion_requirements(self):
        return ["obs"]


class Underlying(ParameterUnderlying):
    def __init__(self):
        super().__init__("under", "$u$")

    def get_suggestion(self, data):
        return 0.5

    def get_suggestion_sigma(self, data):
        return 0.3

    def get_log_prior(self, data):
        return 1


class ToLatent1(Edge):
    def __init__(self):
        super().__init__("obs", "latent1")

    def get_log_likelihood(self, data):
        return np.log(0.1 * np.ones(data["obs"].shape))


class ToTransformation1(EdgeTransformation):
    def __init__(self):
        super().__init__("t1", "latent1")

    def get_transformation(self, data):
        return {"t1": data["latent1"]}


class ToTransformation2(EdgeTransformation):
    def __init__(self):
        super().__init__("t2", ["latent2", "discrete"])

    def get_transformation(self, data):
        return {"t2": data["latent2"]}


class ToDiscrete(Edge):
    def __init__(self):
        super().__init__(["latent1", "t1"], ["discrete", "latent1", "t2"])

    def get_log_likelihood(self, data):
        return np.log(0.1 * np.ones(data["latent1"].shape))


class ToUnderlying(Edge):
    def __init__(self):
        super().__init__("t2", "under")

    def get_log_likelihood(self, data):
        return np.log(0.1 * np.ones(data["t2"].shape))


class DiscreteModelSecondary(Model):
    def __init__(self):
        super().__init__("DiscreteModel")
        data = np.array([0.0, 0.5])
        self.raw_data = data
        self.add_node(ObservedValue(data))
        self.add_node(Transformed1())
        self.add_node(LatentValue1(data.size))
        self.add_node(DiscreteValue1())
        self.add_node(LatentValue2(data.size))
        self.add_node(Transformed2())
        self.add_node(Underlying())

        self.add_edge(ToLatent1())
        self.add_edge(ToTransformation1())
        self.add_edge(ToUnderlying())
        self.add_edge(ToDiscrete())
        self.add_edge(ToTransformation2())

        self.finalise()


class DiscreteModelSecondary2(Model):
    def __init__(self):
        super().__init__("DiscreteModel")
        data = np.array([0.0, 0.5])
        self.raw_data = data
        self.add_node(ObservedValue(data))
        self.add_node(Transformed1())
        self.add_node(LatentValue1(data.size))
        self.add_node(DiscreteValue2())
        self.add_node(LatentValue2(data.size))
        self.add_node(Transformed2())
        self.add_node(Underlying())

        self.add_edge(ToLatent1())
        self.add_edge(ToTransformation1())
        self.add_edge(ToUnderlying())
        self.add_edge(ToDiscrete())
        self.add_edge(ToTransformation2())

        self.finalise()


class TestDiscreteSecondary(object):
    model = DiscreteModelSecondary()
    theta = [0.6, 0.5, 0.5, 0.5, 0.5]

    def test_latent_num_parameters(self):
        assert len(self.model._theta_names) == 5

    def test_latent_prior(self):
        assert self.model.get_log_prior(self.theta) == 1.0

    def test_latent_posterior(self):
        posterior = np.log((0.1 * (0.1 * 0.1 + 0.1 * 0.1))**2)
        posterior += 1
        model_posterior = self.model.get_log_posterior(self.theta)
        assert np.isclose(model_posterior, posterior)


class TestDiscreteSecondary2(object):
    model = DiscreteModelSecondary2()
    theta = [0.6, 0.5, 0.5, 0.5, 0.5]

    def test_latent_num_parameters(self):
        assert len(self.model._theta_names) == 5

    def test_latent_prior(self):
        assert self.model.get_log_prior(self.theta) == 1.0

    def test_latent_posterior(self):
        posterior = np.log((0.1 * (0.1 * 0.1 + 0.1 * 0.1))**2)
        posterior += 1
        model_posterior = self.model.get_log_posterior(self.theta)
        assert np.isclose(model_posterior, posterior)
