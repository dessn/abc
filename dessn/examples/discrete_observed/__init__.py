r""" An example used to prototype the inclusion of discrete observables into the mix.

The scenario being modelled is similar to the coloured balled example. Except now
we introduce a measurement deficiency. 20% of the time when we measure the colour of
the balls, we fumble. On a fumble, we have a 30% chance of writing down the
wrong colour.

We represent this by giving colour as a discrete parameter, where 80% of the time
there is only one colour option, but 20% of the time we give both colours as options,
quantifying the probability that we switched the colours or not.

----------

As normal, the framework is set up by declaring parameters (which can be thought of like
nodes on a PGM), and declaring the edges between parameters (the conditional probabilities).

This is the primary class in this package, and you can see that other classes
inherit from either :class:`.Parameter` or from :class:`.Edge`.

I leave the documentation for :class:`.Parameter` and :class:`.Edge` to those classes,
and encourage viewing the code directly to understand exactly what is happening.

Running this file in python first generates a PGM of the framework, and then runs ``emcee``
and creates a corner plot:

.. figure::     ../dessn/examples/discrete_observed/output/pgm.png
    :align:     center

.. figure::     ../dessn/examples/discrete_observed/output/surfaces.png
    :align:     center

"""