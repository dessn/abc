import numpy as np
import pickle
import os
from dessn.chain.chain import ChainConsumer

dir_name = os.path.dirname(__file__)
c = ChainConsumer()

for i in range(10):
    t = os.path.abspath(dir_name + "/output/temp%d.pkl" % i)
    with open(t, 'rb') as output:
        chain = pickle.load(output)
        chain = np.vstack((chain["mu"], chain["sigma"]))

    c.add_chain(chain.T, parameters=[r"$\mu$", r"$\sigma$"])
c.configure_bar(shade=True)
c.configure_contour(sigmas=[0, 0.01, 1, 2], contourf=True, contourf_alpha=0.1)
c.plot(display=True, truth=[100, 20])