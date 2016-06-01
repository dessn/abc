import numpy as np
from dessn.chain.chain import ChainConsumer


mean = np.array([0.0, 4.0])
cov = np.array([[1.0, 0.7], [0.7, 1.5]])
data = np.random.multivariate_normal(mean, cov, size=100000)

c = ChainConsumer()
c.add_chain(data, parameters=["$x_1$", "$x_2$"])
c.plot(filename="demoIntroduction.png", figsize="column", truth=[0.0, 4.0])
