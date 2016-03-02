import numpy as np
import emcee
import matplotlib.pyplot as plt
from progressbar import Bar, Counter, ETA, Percentage, ProgressBar, Timer
import corner
import os
from example import Example

    
class ExampleLatentClass(Example):
    r"""An implementation of :class:`.ExampleLatent` using classes instead of procedural code.

    Parameters
    ----------
    n : int, optional
        The number of supernova to 'observe'
    theta_1 : float, optional
        The mean of the underlying supernova luminosity distribution
    theta_2 : float, optional
        The standard deviation of the underlying supernova luminosity distribution
    """

    def __init__(self, n=300, theta_1=100.0, theta_2=30.0):
        super(ExampleLatentClass, self).__init__(n, theta_1, theta_2)

    def get_likelihood(self, theta, data, error):
        return -np.inf

    def do_emcee(self, nwalkers=None, nburn=None, nsteps=None):
        pass


