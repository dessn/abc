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
    Z_i &\sim \mathcal{N}(\hat{Z}_i, \hat{C}) \\
    c_i &= 10^{\hat{Z}_i / 2.5} f \\
    \mathbf{\hat{c}_i} &\sim \mathcal{N}(c_i, \sqrt{c_i})

The model PGM is constructed as follows:

.. figure::     ../dessn/proofs/efficiency_6/output/pgm.png
    :width:     80%
    :align:     center
"""