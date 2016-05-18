r""" An example used to prototype the inclusion of discrete observables into the mix.

The scenario being modelled is similar to the coloured balled example. Except now
we introduce a measurement deficiency. 20% of the time when we measure the colour of
the balls, we fumble. On a fumble, we have a 30% chance of writing down the
wrong colour.

We represent this by giving colour as a discrete parameter, where 80% of the time
there is only one colour option, but 20% of the time we give both colours as options,
quantifying the probability that we switched the colours or not.

"""