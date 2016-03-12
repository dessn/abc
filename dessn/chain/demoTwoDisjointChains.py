import numpy as np
from chain import ChainConsumer


class DemoTwoDisjointChains:
    r""" The multiple chain demo for Chain Consumer. Dummy class used to get documentation caught by ``sphinx-apidoc``,
    it servers no other purpose.

    Running this file in python creates two random data sets, representing two separate chains, *for two separate models*.

    It is sometimes the case that we wish to compare models which have partially overlapping parameters. For example,
    we might fit a model which depends has cosmology dependend on :math:`\Omega_m` and :math:`\Omega_\Lambda`, where we
    assume :math:`w = 1`. Alternatively, we might assume flatness, and therefore fix :math:`\Omega_\Lambda` but instead
    vary the equation of state :math:`w`. The good news is, you can visualise them both at once!


    The second thing we do is create a consumer, and load both chains into it. As we have different parameters for each
    chain we supply the right parameters for each chain. The plot for this is saved to the png file below:

    .. figure::     ../dessn/chain/demoTwoDisjointChains.png
        :align:     center

    """
    def __init__(self):
        pass

if __name__ == "__main__":
    ndim, nsamples = 4, 200000
    np.random.seed(0)

    data = np.random.randn(nsamples, ndim)
    data[:, 2] += data[:, 1] * data[:, 2]
    data[:, 1] = data[:, 1] * 3 + 5
    data[:, 3] /= (np.abs(data[:, 1]) + 1)

    data2 = np.random.randn(nsamples, ndim)
    data2[:, 0] -= 1
    data2[:, 2] += data2[:, 1]**2
    data2[:, 1] = data2[:, 1] * 2 - 5
    data2[:, 3] = data2[:, 3] * 1.5 + 2

    # If you pass in parameter labels and only one chain, you can also get parameter bounds
    c = ChainConsumer()
    c.add_chain(data, parameters=["$x$", "$y$", r"$\alpha$", r"$\beta$"])
    c.add_chain(data2, parameters=["$x$", "$y$", r"$\alpha$", r"$\gamma$"])
    c.plot(figsize="page", display=True, filename="demoTwoDisjointChains.png")
