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
    \mu &\sim \mathcal{U}(500, 1500) \\
    \sigma &\sim \mathcal{U}(50, 150) \\
    Z_i &\sim \mathcal{N}(Z_o, C) \\
    L &\sim \mathcal{N}(\mu, \sigma) \\
    z &\sim \mathcal{U}(0.5, 1.5) \\
    f &= \frac{L}{z^2} \\
    c_i &= 10^{Z_i / 2.5} f \\
    \mathbf{\hat{c}_i} &\sim \mathcal{N}(c_i, \sqrt{c_i})

We create a data set by drawing from these distributions and introducing our
data selection cuts. For 400 events, this gives us the following data distribution
in redshift and luminosity.

.. figure::     ../dessn/proofs/efficiency_6/output/data.png
    :width:     80%
    :align:     center



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
denominator. Unlike in the previous example, stipulating at least two points
above a signal to noise of :math:`\alpha` in *any* bands can be trivially broken into
a cut of :math:`1 - P(S_0|\mu,\sigma,Z_i) - P(S_1|\mu,\sigma,Z_i)` because of the
combinatorics involved now. However, we can determine :math:`P(S_0|\mu,\sigma,Z_i` for
each band and combine these to get the correct weights. We consider the case
where we have :math:`N` observations in each band and looking at band 0 to
begin with.

.. math::
    P(S_0|\mu,\sigma,Z_0) &= P(S_0|\mu,\sigma,Z_0)

To determine this value, we introduce integrals. Lots of integrals. Too many, some would say.

.. math::
    P(S_0|\mu,\sigma,Z_0) &= \int dL \int dc \int dz\  P(S_0, c, z, L | \mu, \sigma, Z_0) \\
    &= \int dz\ z^2 P(z) \int df\ P(f z^{-2}|\mu,\sigma) \int dc\
    P(S_0 | \mathbf{c}) P(\mathbf{c} | L, z, Z_0) \\
    &= \int dz\ z^2 P(z) \int df\ P(f z^{-2}|\mu,\sigma) \prod_{N} \int_{-\infty}^{\alpha^2} dc\ \mathcal{N}\left(c ; 10^{Z_0/2.5}f, \sqrt{10^{Z_0/2.5}f} \right)

In the last example we showed that

.. math::
    \int_{-\infty}^{\alpha^2} dc\ \mathcal{N}\left(c ; 10^{Z_0/2.5}f, \sqrt{10^{Z_0/2.5}f} \right)
    &= \begin{cases}
    \frac{1}{2} + \frac{1}{2}{\rm erf} \left[ \frac{\alpha^2 - 10^{Z_0/2.5}f}{\sqrt{2f\cdot 10^{Z_0/2.5}}} \right] &
    \text{ if } \alpha^2 - 10^{Z_0/2.5}f > 0 \\
    \frac{1}{2} - \frac{1}{2}{\rm erf} \left[ \frac{10^{Z_0/2.5}f - \alpha^2}{\sqrt{2f\cdot 10^{Z_0/2.5}}} \right] &
    \text{ if } \alpha^2 - 10^{Z_0/2.5}f < 0 \\
    \end{cases} \\
    &= g_{-}(f,Z_0,\alpha)

Substituting this in, along with a flat distribution (of width 1) on redshift and
the normal distribution on luminosity, we get

.. math::
    P(S_0|\mu,\sigma,Z_0) = \int dz\ z^2 \int df\ \mathcal{N}(f z^{-2};\mu,\sigma)
    \left[g_{-}(f,Z_0,\alpha)\right]^N

From the previous example, we can also see that

.. math::
    P(S_1|\mu,\sigma,Z_0) &= \int dz\ z^2  \int df \ \mathcal{N}(fz^{-2};\mu,\sigma)
    N  g_{+}(f,\alpha, Z_0) \left[g_{-}(f,\alpha, Z_0)\right]^{N-1}\\

In order to combine these into the actual model efficiency, we need to permutate them.
For two bands, this is given by

.. math::
    P(S_2|\mu,\sigma,Z_0) = 1 -
    \sum_{i\in \lbrace0,1\rbrace} \sum_{j\in \lbrace 0, 1 \rbrace}
    P(S_i|\mu,\sigma,Z_0) P(S_j|\mu,\sigma,Z_1)


The model PGM is constructed as follows:

.. figure::     ../dessn/proofs/efficiency_6/output/pgm.png
    :width:     100%
    :align:     center


The weights (efficiency, denominator, or mathematically :math:`P(S_2|\mu,\sigma,Z_i)`),
sliced down a single value of :math:`Z_0` and :math:`Z_1`, appear as follows:

.. figure::     ../dessn/proofs/efficiency_6/output/weights.png
    :width:     80%
    :align:     center


Here are plotted the likelihood surfaces. Green represents all the data, including those
that didn'nt make the cut, using a model without a bias correction. Red is the data
after the cut (less data points), using a model without the bias correction. Blue is with the
biased data, using a model implementing the bias correction.

.. figure::     ../dessn/proofs/efficiency_6/output/surfaces.png
    :align:     center
    :width:     100%
"""