import numpy as np
import emcee
from scipy import stats
import matplotlib.pyplot as plt
from progressbar import Bar, Counter, ETA, Percentage, ProgressBar, Timer
import corner
import os
import abc

class Example(object):
    r"""An example implementation using integration over a latent parameter.

    Let us assume that we are observing supernova that a drawn from an underlying
    supernova distribution parameterised by :math:`\theta`,
    where the supernova itself simply a luminosity :math:`L`. We measure the luminosity
    of multiple supernovas, giving us an array of measurements :math:`D`. If we wish to recover
    the underlying distribution of supernovas from our measurements, we wish to find :math:`P(\theta|D)`,
    which is given by

    .. math::
        P(\theta|D) \propto P(D|\theta)P(\theta)


    Note that in the above equation, we realise that :math:`P(D|L) = \prod_{i=1}^N P(D_i|L_i)` as our measurements are
    independent. The likelihood :math:`P(D|\theta)` is given by

    .. math::
        P(D|\theta) =  \prod_{i=1}^N  \int_{-\infty}^\infty P(D_i|L_i) P(L_i|\theta) dL_i



    We now have two distributions to characterise. Let us assume both are gaussian, that is
    our observed luminosity :math:`x_i` has gaussian error :math:`\sigma_i` from the actual supernova
    luminosity, and the supernova luminosity is drawn from an underlying gaussian distribution
    parameterised by :math:`\theta`.

     .. math::
        P(D_i|L_i) = \frac{1}{\sqrt{2\pi}\sigma_i}\exp\left(-\frac{(x_i-L_i)^2}{\sigma_i^2}\right)

        P(L_i|\theta) = \frac{1}{\sqrt{2\pi}\theta_2}\exp\left(-\frac{(L_i-\theta_1)^2}{\theta_2^2}\right)



    This gives us a likelihood of

    .. math::

        P(D|\theta) = \prod_{i=1}^N  \frac{1}{2\pi \theta_2 \sigma_i}  \int_{-\infty}^\infty
        \exp\left(-\frac{(x_i-L_i)^2}{\sigma_i^2} -\frac{(L_i-\theta_1)^2}{\theta_2^2} \right) dL_i


    Working in log space for as much as possible will assist in numerical precision, so we can rewrite this as

    .. math::
        \log\left(P(D|\theta)\right) =  \sum_{i=1}^N  \left[
                \log\left( \int_{-\infty}^\infty \exp\left(-\frac{(x_i-L_i)^2}{\sigma_i^2} -
        \frac{(L_i-\theta_1)^2}{\theta_2^2} \right) dL_i \right) -\log(2\pi\theta_2\sigma_i) \right]
        
        
    Parameters
    ----------
    n : int, optional
        The number of supernova to 'observe'
    theta_1 : float, optional
        The mean of the underlying supernova luminosity distribution
    theta_2 : float, optional
        The standard deviation of the underlying supernova luminosity distribution

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, n=300, theta_1=100.0, theta_2=30.0):
        self.n = n
        self.theta_1 = theta_1
        self.theta_2 = theta_2

        self.data = stats.norm.rvs(size=n, loc=theta_1, scale=theta_2)
        self.error = 0.2 * np.sqrt(self.data)
        self.data += stats.norm.rvs(size=n) * self.error

        self.sampler = None
        self.sample = None

    def plot_observations(self):
        """Plot the observations and observation distribution.
        """
        fig, ax = plt.subplots(figsize=(6, 4), ncols=2)
        x = np.arange(self.n)
        ax[0].errorbar(x, self.data, yerr=self.error, fmt='o')
        ax[1].hist(self.data, bins=10, histtype='step', normed=True)

        xs = np.linspace(self.data.min(), self.data.max(), 100)
        ys = (1 / (np.sqrt(2 * np.pi) * self.theta_2)) * np.exp(-(xs - self.theta_1) ** 2 / (self.theta_2 ** 2))
        ax[1].plot(xs, ys, alpha=0.5)
        plt.show()

    def _integrand(self, l, theta, d, e):
        return -((d - l) * (d - l) / (e * e)) - (l - theta[0]) * (l - theta[0]) / (theta[1] * theta[1])

    def _plus(self, x, y):
        if x > y:
            return x + np.log(1 + np.exp(y - x))
        else:
            return y + np.log(1 + np.exp(x - y))

    @abc.abstractmethod
    def get_likelihood(self, theta, data, error):
        return -np.inf

    def get_prior(self, theta):
        r""" Get the log prior probability given the input.

        The prior distribution is currently implemented as flat prior.

        Parameters
        ----------
        theta : array of model parameters

        Returns
        -------
        float
            the log prior probability
        """
        if theta[0] < 0 or theta[0] > 200 or theta[1] < 0 or theta[1] > 50:
            return -np.inf
        else:
            return 1

    def get_posterior(self, theta, data, error):
        r""" Gives the log posterior probability given the supplied input parameters.

        Parameters
        ----------
        theta : array of model parameters
        data : array of length `n`
            An array of observed luminosities
        error : array of length `n`
            An array of observed luminosity errors

        Returns
        -------
        float
            the log posterior probability
        """
        prior = self.get_prior(theta)
        if np.isfinite(prior):
            likelihood = self.get_likelihood(theta, data, error)
            return likelihood + prior
        else:
            return prior

    @abc.abstractmethod
    def do_emcee(self, nwalkers=None, nburn=None, nsteps=None):
        pass
