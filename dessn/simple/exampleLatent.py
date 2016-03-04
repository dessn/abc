import numpy as np
from dessn.simple.example import Example

    
class ExampleLatent(Example):
    r"""An example implementation using marginalisation over latent parameters.

    Building off the math from :class:`.Example`, instead of performing the integration numerically in the computation of the likelihood, we can
    instead use Monte Carlo integration by simply setting the latent parameters :math:`\vec{L}` as 
    free parameters, giving us
    
    .. math::
        \log\left(P(D|\theta,\vec{L})\right) = - \sum_{i=1}^N  \left[
                \frac{(x_i-L_i)^2}{\sigma_i^2} +
        \frac{(L_i-\theta_1)^2}{\theta_2^2} + \log(2\pi\theta_2\sigma_i) \right]
        
    Creating this class will set up observations from an underlying distribution.
    Invoke ``emcee`` by calling the object. In this example, we marginalise over :math:`L_i` after
    running our MCMC, and so we no longer have to compute integrals in our chain, but instead have
    dimensionality of :math:`2 + n`, where :math:`n` are the number of observations.
    
    
    .. figure::     ../plots/exampleLatent.png
        :align:     center


    Parameters
    ----------
    n : int, optional
        The number of supernova to 'observe'
    theta_1 : float, optional
        The mean of the underlying supernova luminosity distribution
    theta_2 : float, optional
        The standard deviation of the underlying supernova luminosity distribution
    """

    def __init__(self, n=30, theta_1=100.0, theta_2=20.0):
        super(ExampleLatent, self).__init__(n, theta_1, theta_2)

    def get_likelihood(self, theta, data, error):
        r""" Gets the log likelihood given the supplied input parameters.

        Parameters
        ----------
        theta : array of length 2 + :math:`n`
            An array representing :math:`[\theta_1,\theta_2,\vec{L}]`
        data : array of length :math:`n`
            An array of observed luminosities
        error : array of length :math:`n`
            An array of observed luminosity errors

        Returns
        -------
        float
            the log likelihood probability
        """
        result = 0
        ls = theta[2:]
        for l, d, e in zip(ls, data, error):
            result -= np.log(2 * np.pi * e * theta[1])
            result += self._integrand(l, theta, d, e)

        if not np.isfinite(result):
            return -np.inf
        return result

    def do_emcee(self, nwalkers=500, nburn=2000, nsteps=2500):
        """ Run the `emcee` chain and produce a corner plot.

        Saves a png image of the corner plot to plots/exampleLatent.png.

        Parameters
        ----------
        nwalkers : int, optional
            The number of walkers to use.
        nburn : int, optional
            The burn in period of the chains.
        nsteps : int, optional
            The number of steps to run
        """
        ndim = 2 + self.n
        ndim_final = 2
        starting_guesses = np.random.normal(1, 0.2, (nwalkers, ndim))
        starting_guesses[:, 0] *= self.theta_1
        starting_guesses[:, 1] *= self.theta_2
        starting_guesses[:, 2:] *= self.theta_1

        self._run_emcee(ndim, nburn, nsteps, nwalkers, starting_guesses, ndim_final, "exampleLatent")


if __name__ == "__main__":
    example = ExampleLatent()
    # example.plot_observations()
    example.do_emcee()
