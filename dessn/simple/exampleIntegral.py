import numpy as np
from dessn.simple.example import Example


class ExampleIntegral(Example):
    r"""An example implementation using integration over a latent parameter.

    Building off the math from :class:`.Example`
    Creating this class will set up observations from an underlying distribution.
    Invoke ``emcee`` by calling the object. In this example, we perform the marginalisation
    inside the likelihood calculation, which gives us dimensionality only of two (the length of
    the :math:`\theta` array). However, this is at the expense of  performing the marginalisation over
    :math:`dL_i`, as this requires computing :math:`n` integrals for each step in the MCMC.

    .. figure::     ../plots/exampleIntegration.png
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

    def __init__(self, n=10, theta_1=100.0, theta_2=30.0):
        super(ExampleIntegral, self).__init__(n, theta_1, theta_2)


    def _integrate(self, d, e, theta):
        step = np.linspace(0, 200, 400)
        diff = step[1] - step[0]
        r = self._integrand(step[0], theta, d, e) - diff
        for s in step[1:]:
            r = self._plus(r, self._integrand(s, theta, d, e) - diff)
        return r

    def get_likelihood(self, theta, data, error):
        r""" Gets the log likelihood given the supplied input parameters.

        Parameters
        ----------
        theta : array of size 2
            An array representing :math:`[\theta_1,\theta_2]`
        data : array of length `n`
            An array of observed luminosities
        error : array of length `n`
            An array of observed luminosity errors

        Returns
        -------
        float
            the log likelihood probability
        """
        result = 0
        for d, e in zip(data, error):
            result -= np.log(2 * np.pi * e * theta[1])
            result += self._integrate(d, e, theta)

        if not np.isfinite(result):
            return -np.inf
        return result

    def do_emcee(self, nwalkers=20, nburn=2500, nsteps=3000):
        """ Run the `emcee` chain and produce a corner plot.

        Saves a png image of the corner plot to plots/exampleIntegration.png.

        Parameters
        ----------
        nwalkers : int, optional
            The number of walkers to use. Minimum of four.
        nburn : int, optional
            The burn in period of the chains.
        nsteps : int, optional
            The number of steps to run
        """
        ndim = 2
        starting_guesses = np.random.normal(1, 0.2, (nwalkers, ndim))
        starting_guesses[:, 0] *= self.theta_1
        starting_guesses[:, 1] *= self.theta_2

        self._run_emcee(ndim, nburn, nsteps, nwalkers, starting_guesses, ndim, "exampleIntegration")

if __name__ == "__main__":
    example = ExampleIntegral()
    # example.plot_observations()
    example.do_emcee()

