r""" In this example we add in multiple bands and flux calibration.

As before, we observe "supernova" from an underlying distribution, and the
absolute luminosity to apparent luminosity via some (different) distance
relation, which we shall cringe and denote redshift. The observed zero points and covariance,
denoted :math:`\hat{Z}` and :math:`\hat{C}`, allow us to estimate the actual zero points :math:`Z`,
and using the zero points and apparent magnitude (ie the flux), we can predict counts
for each band (which I will treat as an observed quantity). Using then the predicted
photon counts, which also gives us the error via the Poisson process, we can calculate the
likelihood of our observaitons. We assume the same apparent magnitude in all bands,
for clarification.

As I now use subscripts to denote different bands, with :math:`i` representing
all different bands (like Einstein notation), and so observed quantities will
be represented by a hat and vector quantities shall be represented by bold font.

.. math::
    L &\sim \mathcal{N}(\mu, \sigma) \\
    z &\sim \mathcal{U}(0.5, 1.5) \\
    f &= \frac{L}{z^2} \\
    Z_i &\sim \mathcal{N}(\hat{Z}_i, C) \\
    c_i &= 10^{\hat{Z}_i / 2.5} f \\
    \mathbf{\hat{c}_i} &\sim \mathcal{N}(c_i, \sqrt{c_i})

Denoting the selection effects this time as :math:`S`, and implementing flat priors, we
have:

.. math::
    \mathcal{L} &= P(\mathbf{\hat{c}_i}, \hat{z}, \hat{Z}_i|S,\mu,\sigma) \\
    &= \frac{P(\mathbf{\hat{c}_i}, \hat{z}, \hat{Z}_i, S|\mu,\sigma)}{P(S|\mu,\sigma)} \\
    &= \frac{\int dZ_i \int dL P(\mathbf{\hat{c}_i}, \hat{z}, \hat{Z}_i, S, L, Z_i|\mu,\sigma)}{P(S|\mu,\sigma)} \\
    &= \frac{\int dZ_i \int dL \ P(S | \mathbf{\hat{c}_i}) P(\mathbf{\hat{c}_i}| L, Z_i, \hat{z}) P(\hat{Z}_i | Z_i)
    P(\hat{z}) P(L|\mu,\sigma)}{P(S|\mu,\sigma)}

Here we also assume flat priors on redshift, and as our
data has already gone through the selection cuts, :math:`P(S|\mathbf{\hat{c}_i}) = 1`.

.. math::
    \mathcal{L} &= \frac{\int dZ_i \int dL \  P(\mathbf{\hat{c}_i}| L, Z_i, \hat{z})
    P(\hat{Z} |  Z_i) P(L|\mu,\sigma)}{P(S|\mu,\sigma)} \\

Transforming from luminosity to flux using the relation :math:`f = \frac{L}{\hat{z}^2}`, which
gives us :math:`dL = \hat{z}^2 df`:

.. math::
    \mathcal{L} &= \frac{\int dZ_i \int df \ \hat{z}^2 \  P(\mathbf{\hat{c}_i}| f, Z_i)
    P(\hat{Z}_i | Z_i) P(\frac{f}{\hat{z}^2}|\mu,\sigma)}{P(S|\mu,\sigma)} \\
    &= \frac{\int dZ_i \int df \ \hat{z}^2 \
    \mathcal{N}\left(\mathbf{\hat{c}_i}; 10^{Z_i/2.5} f, \sqrt{10^{Z_i/2.5}f}\right)
    \mathcal{N}(\hat{Z}_i ; Z_i, \hat{C})
    \mathcal{N}\left(\frac{f}{\hat{z}^2} ; \mu, \sigma\right)}   {P(S|\mu,\sigma)} \\

We now need to go through the fun process of determining the model efficiency in the
denominator. As in the previous example, stipulating at least two points
above a signal to noise of :math:`\alpha` in *all* bands can be broken into
a cut of :math:`1 - P(S_0|\mu,\sigma) - P(S_1|\mu,\sigma)`. To start with, we can
examine the case of no observations above the cut in all bands.

To do so, we introduce integrals. Lots of integrals. Too many, some would say.

.. math::
    P(S|\mu,\sigma) &=


The model PGM is constructed as follows:

.. figure::     ../dessn/proofs/efficiency_6/output/pgm.png
    :width:     100%
    :align:     center



"""