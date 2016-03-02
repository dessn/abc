""" This module is designed to give a step by step overview of a very simplified example
Bayesian model.

The basic example model is laid out in the parent class :class:`.Example`,
and there are three implementations. The first implementation, :class:`.ExampleIntegral`, shows
how the problem might be approached in a simple model, where numerical integration is simply done
as part of the likelihood calculation.

However, if there are multiple latent parameters, we get polynomial growth of the number of numerical
integrations we have to do, and so this does not scale well at all.

This leads us to the implementation in :class:`.ExampleLatent`, where we use the MCMC algorithm to
essentially do Monte Carlo integration via marginalisation. This means we do not need to perform
the numerical integration in the likelihood calculation, however the cost of doing so is increase
dimensionality of our MCMC.

Finally, the :class:`.ExampleLatentClass` implementation shows how the :class:`.ExampleLatent` class
might be written to make use of Nodes. This is done in preparation for more complicated models, which will
have more than one layer and needs to be configurable.
"""