r""" In this example we add in multiple bands and flux calibration.

As before, we observe "supernova" from an underlying distribution, and the
absolute luminosity to apparent luminosity via some (different) distance
relation, which we shall cringe and denote redshift. The underlying zero points :math:`Z_i` are
informed by a strong calibration prior on zero points :math:`Z_o` and covariance :math:`C`.
Using the zero points and apparent magnitude (ie the flux), we can predict counts
for each band (which I will treat as an observed quantity). Using then the predicted
photon counts, which also gives us the error via the Poisson process, we can calculate the
likelihood of our observations. We assume the same apparent magnitude in all bands,
for clarification.

As I now use subscripts to denote different bands, with :math:`i` representing
all different bands (like Einstein notation), and so observed quantities will
be represented by a hat and vector quantities shall be represented by bold font.

.. math::
    \mu &\sim \mathcal{U}(0, 1000) \\
    \sigma &\sim \mathcal{U}(0, 100) \\
    Z_i &\sim \mathcal{N}(Z_o, C) \\
    L &\sim \mathcal{N}(\mu, \sigma) \\
    z &\sim \mathcal{U}(0.5, 1.5) \\
    f &= \frac{L}{z^2} \\
    c_i &= 10^{Z_i / 2.5} f \\
    \mathbf{\hat{c}_i} &\sim \mathcal{N}(c_i, \sqrt{c_i})

Denoting the selection effects - a signal to noise cut in all bands - this
time as :math:`S_2` we have:

.. math::
    \mathcal{L} &= P(\mathbf{\hat{c}_i}, \hat{z}|S_2,\mu,\sigma, Z_i) \\
    &= \frac{P(\mathbf{\hat{c}_i}, \hat{z}, S_2|\mu,\sigma, Z_i)}{P(S_2|\mu,\sigma, Z_i)} \\
    &= \frac{\int dL P(\mathbf{\hat{c}_i}, \hat{z}, S_2, L|\mu,\sigma, Z_i)}{P(S_2|\mu,\sigma, Z_i)} \\
    &= \frac{\int dL \ P(S_2 | \mathbf{\hat{c}_i}) P(\mathbf{\hat{c}_i}| L \hat{z})
    P(\hat{z}) P(L|\mu,\sigma)}{P(S_2|\mu,\sigma, Z_i)}

Here we also assume flat priors on redshift, and as our
data has already gone through the selection cuts, :math:`P(S_2|\mathbf{\hat{c}_i}) = 1`.

.. math::
    \mathcal{L} &= \frac{\int dL \  P(\mathbf{\hat{c}_i}| L, Z_i, \hat{z})
    P(L|\mu,\sigma)}{P(S_2|\mu,\sigma, Z_i)} \\
    &= \frac{\int dL \  P(\mathbf{\hat{c}_i}| L, \hat{z}, Z_i)
    P(L|\mu,\sigma)}{P(S_2|\mu,\sigma, Z_i)} \\
    &= \frac{\int dL \ \mathcal{N}\left(\mathbf{\hat{c}_i}; 10^{Z_i/2.5} \frac{L}{\hat{z}^2}, \sqrt{10^{Z_i/2.5}\frac{L}{\hat{z}^2}}\right)
    \mathcal{N}\left(L ; \mu, \sigma\right)}   {P(S_2|\mu,\sigma,Z_i)} \\

Finally, as flux is easier to estimate than luminosity for a starting position,
we transform our integral over luminosity into an integral over flux.
Which gives us :math:`dL = \hat{z}^2 df`, leading to:

.. math::
    \mathcal{L} &= \frac{\int df\ \hat{z}^2 \ \mathcal{N}\left(\mathbf{\hat{c}_i}; 10^{Z_i/2.5} f, \sqrt{10^{Z_i/2.5}f}\right)
    \mathcal{N}\left(\frac{f}{\hat{z}^2} ; \mu, \sigma\right)}   {P(S_2|\mu,\sigma,Z_i)} \\

We now need to go through the fun process of determining the model efficiency in the
denominator. As in the previous example, stipulating at least two points
above a signal to noise of :math:`\alpha` in *all* bands can be broken into
a cut of :math:`1 - P(S_0|\mu,\sigma,Z_i) - P(S_1|\mu,\sigma,Z_i)`. To start with, we can
examine the case of no observations above the cut in all bands, which is a product of
the probability of no observations above the cut in each separate band.

.. math::
    P(S_0|\mu,\sigma,Z_i) &= \prod_b P(S_0|\mu,\sigma,Z_{i=b})

Let's look it the first band :math:`Z_0`. To do so, we introduce integrals.
Lots of integrals. Too many, some would say.

.. math::
    P(S_0|\mu,\sigma,Z_0) &= \int dL \int dc \int dz\  P(S_0, c, z, L | \mu, \sigma, Z_0) \\
    &= \int dz P(z) \int dL\ P(L|\mu,\sigma) \int dc\ P(S_0 | c) P(c | L, z, Z_0)


The model PGM is constructed as follows:

.. figure::     ../dessn/proofs/efficiency_6/output/pgm.png
    :width:     100%
    :align:     center


Here are plotted the likelihood surfaces. Green represents all the data, including those
that didn'nt make the cut, using a model without a bias correction. Red is the data
after the cut (less data points), using a model without the bias correction. Blue is with the
biased data, using a model implementing the bias correction.

.. figure::     ../dessn/proofs/efficiency_6/output/surfaces.png
    :align:     center
    :width:     100%
"""