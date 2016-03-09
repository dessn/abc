import numpy as np
from chain import ChainConsumer


if __name__ == "__main__":
    ndim, nsamples = 3, 200000

    data = np.random.randn(nsamples, ndim)
    data[:, 2] += data[:, 1] * data[:, 2]
    data[:, 1] = data[:, 1] * 3 + 5

    # You can plot the data directly without worrying about labels
    # Code follows the fluent pattern, so you can string calls together
    ChainConsumer().add_chain(data).plot()

    # If you pass in parameter labels and only one chain, you can also get parameter bounds
    ChainConsumer().add_chain(data, parameters=["$x$", "$y$", r"$\epsilon$"], name="Test chain").plot(filename="demoSingle.png")
