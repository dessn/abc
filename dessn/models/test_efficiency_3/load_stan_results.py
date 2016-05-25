import numpy as np
from dessn.chain.chain import ChainConsumer
import pickle
import os
from dessn.models.test_efficiency_3.run_stan import get_data

m, s, d, e, a, n = get_data()

dir_name = os.path.dirname(__file__)
t = os.path.abspath(dir_name + "/output/temp.pkl")
with open(t, 'rb') as output:
    chain = pickle.load(output)
    chain = np.vstack((chain["mu"][1000:], chain["sigma"][1000:],
                       chain["s"][1000:, 0], chain["s"][1000:, 1], chain["s"][1000:, 2],
                       chain["s"][1000:, 3], chain["s"][1000:, 4], chain["s"][1000:, 5]))

c = ChainConsumer()
c.add_chain(chain.T, parameters=[r"$\mu$", r"$\sigma$", "s1", "s2", "s3", "s4", "s5", 's6'])
c.plot(display=True, truth=[100, 20] + d[:6].tolist())
