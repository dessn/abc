r"""

To increase complexity, in this example we have multiple observations of the flux,
and our selection criteria will be at least two points are above a signal to noise
threshold. Formally,

.. math::
    L &\sim \mathcal{N}(\mu, \sigma) \\
    z &\sim \mathcal{U}(0, 1.5) \\
    f &= \frac{L}{(1+z)^2} \\
    f_o &\sim \mathcal{N}(f, \sqrt{f})

As before, when observing we approximate the error on :math:`f_o` as :math:`\sqrt{f_o}`,
and so our signal to noise cut of :math:`\alpha` is given by :math:`f_o > \alpha^2`.

As in the previous examples, we implement flat priors such that our posterior surface
becomes proportional to our likelihood surface. We denote our sample selection cuts by the
general parameter :math:`C_2`, so our likelihood is then given by

.. math::
    \mathcal{L} &= \frac{P(\vec{f_o}, z_o, C_2 |\mu, \sigma)}{P(C_2|\mu,\sigma)} \\
    &= \frac{\int dL \int dz P(C_2, \vec{f_o}, z_o, z, L| \mu, \sigma) P(z)}{P(C_2|\mu,\sigma)} \\
    &= \frac{\int df\ P(C_2|\vec{f_o}) P(\vec{f_o}|f) (1+z_o)^2 P(f(1+z_o)^2| \mu, \sigma) P(z_o)}{P(C_2|\mu,\sigma)} \\
    &= \frac{\int df\  (1+z_o)^2 \mathcal{N}(f(1+z_o)^2 ; \mu, \sigma) P(z_o) \prod_i \mathcal{N}(f_i ; f, \sqrt{f})}{P(C_2|\mu,\sigma)} \\

Where we again translate luminosity into flux.

Once again we have to determine the normalisation term :math:`P(C|\mu,\sigma)`. The probability
of getting two or more points above a signal to noise cutoff is one minus the probability of
getting less than two points above the signal to noise cutoff. One way to write this might be
as such:

.. math::
    P(C_2|\mu,\sigma) = 1 - P(C_0|\mu,\sigma) - P(C_1|\mu,\sigma)

Addressing each term individually, denoting realised flux as :math:`R`...

.. math::
    P(C_0|\mu,\sigma) &= \int dz \int dL \int d\vec{R} \ P(C_0,z,L,\vec{R}|\mu,\sigma) P(z) \\
    &= \int dz (1+z)^2 \int df \int d\vec{R} \ P(C_0|\vec{R}) P(\vec{R}|f) P(f(1+z)^2|\mu,\sigma) P(z) \\
    &= \int dz \frac{1}{z_{\rm max} - z_{\rm min}} (1+z)^2 \int df\  \mathcal{N}(f(1+z)^2;\mu,\sigma) \prod_i \int dR_i  \mathcal{H}(\alpha^2 - R_i) \mathcal{N}(R_i;f,\sqrt{f}) \\
    &= \int_{0}^{1.5} dz \frac{(1+z)^2}{z_{\rm max} - z_{\rm min}}  \int_{-\infty}^\infty df\  \mathcal{N}(f(1+z)^2;\mu,\sigma) \prod_i \int_{-\infty}^{\alpha^2} dR_i  \mathcal{N}(R_i;f,\sqrt{f})

As in the previous examples, we can evaluate the last integral and turn it
into an error function. One simplification that we can employ in this model is that, as
the actual flux is the same in each observation, the integrals should all be the same.

.. math::
    \int_{-\infty}^{\alpha^2} dR_i  \mathcal{N}(R_i;f,\sqrt{f})
    &= \int_{-\infty}^{\alpha^2} dR_i \frac{1}{\sqrt{2\pi f}}\exp\left[-\frac{(R_i - f)^2}{2f}\right] \\
    &= \int_{-\infty}^{\alpha^2 - f} dx \frac{1}{\sqrt{2\pi f}}\exp\left[-\frac{x^2}{2f}\right] \\
    &= \begin{cases}
    \frac{1}{2} + \frac{1}{2}{\rm erf} \left[ \frac{\alpha^2 - f}{\sqrt{2f}} \right] &
    \text{ if } \alpha^2 - f > 0 \\
    \frac{1}{2} - \frac{1}{2}{\rm erf} \left[ \frac{f - \alpha^2}{\sqrt{2f}} \right] &
    \text{ if } \alpha^2 - f < 0 \\
    \end{cases} \\
    &= g_{-}(f,\alpha)

Which gives us


.. math::
    P(C_0|\mu,\sigma) =
    \int_{0}^{1.5} dz \frac{(1+z)^2}{z_{\rm max} - z_{\rm min}}  \int_{-\infty}^\infty df\  \mathcal{N}(f(1+z)^2;\mu,\sigma) \left(g_{-}(f,\alpha)\right)^N

We now consider the more complicated term in the denominator, in which only one point makes
the signal to noise cut, and the others fail. Using :math:`g_{-}(f,\alpha)` for the function
where an observation is below the signal to noise cut,
and :math:`g_{+}(f,\alpha) \equiv 1 - g_{-}(f,\alpha)` as when the point is above the
threshold, and having a total number of observations of :math:`N`, we get:


.. math::
    P(C_1|\mu,\sigma) &= \int dz \int dL \int d\vec{R} \ P(C_1,z,L,\vec{R}|\mu,\sigma) P(z)\\
    &= \int dz \frac{(1+z)^2}{z_{\rm max} - z_{\rm min}} \int df \ P(f(1+z)^2|\mu,\sigma) \int d\vec{R} \ P(C_1|\vec{R}) P(\vec{R}|f) \\
    &= \int dz \frac{(1+z)^2}{z_{\rm max} - z_{\rm min}}  \int df \ \mathcal{N}(f(1+z)^2;\mu,\sigma) \left[ \sum_i \int dR_i \mathcal{H}(R_i - \alpha^2) \right. \\
    &\quad\quad \left.\mathcal{N}(R_i;f,\sqrt{f}) \prod_{j\neq i} \int dR_j \mathcal{H}(\alpha^2 - R_j)\mathcal{N}(R_j;f,\sqrt{f}) \right] \\
    &= \int dz \frac{(1+z)^2}{z_{\rm max} - z_{\rm min}}  \int df \ \mathcal{N}(f(1+z)^2;\mu,\sigma) \left[ \sum_i \left( g_{+}(f,\alpha) \prod_{j\neq i} g_{-}(f,\alpha) \right) \right] \\
    &= \int dz \frac{(1+z)^2}{z_{\rm max} - z_{\rm min}}  \int df \ \mathcal{N}(f(1+z)^2;\mu,\sigma) N  g_{+}(f,\alpha) \left[g_{-}(f,\alpha)\right]^{N-1}\\


This gives us a final likelihood which we can implement:

.. math::
    \mathcal{L} &= \frac{\int df\  \frac{(1+z_o)^2}{z_{\rm max} - z_{\rm min}}  \mathcal{N}(f(1+z_o)^2 ; \mu, \sigma)
    \prod_i \mathcal{N}(f_i ; f, \sqrt{f})}{1 - P(C_0|\mu,\sigma) - P(C_1|\mu,\sigma)} \\
    &= \frac{\int df\  \frac{(1+z_o)^2}{z_{\rm max} - z_{\rm min}}  \mathcal{N}(f(1+z_o)^2 ; \mu, \sigma) \prod_i \mathcal{N}(f_i ; f, \sqrt{f})}
    {1 - \int_{0}^{1} dz \frac{(1+z)^2}{z_{\rm max} - z_{\rm min}}  \int_{-\infty}^\infty df\  \mathcal{N}(f(1+z)^2;\mu,\sigma) \left(g_{-}(f,\alpha)\right)^{N-1} \left[ g_{-}(f,\alpha)  + N  g_{+}(f,\alpha)  \right]} \\


We now can model this using generated data and test our model performance. The denominator term,
as a function of :math:`\mu`, :math:`\sigma` can be thought of as a weighting, and
is shown below. Notice the similarity to the weights shown in the
previous example, however the fact we have changed from a linear decrease in flux due to
redshift to the inverse square law lowers our efficiency across the board.


.. figure::     ../dessn/proofs/efficiency_5/output/weights.png
    :width:     80%
    :align:     center

The model PGM is constructed as follows:

.. figure::     ../dessn/proofs/efficiency_5/output/pgm.png
    :width:     80%
    :align:     center


I plot three realisations of the data uaing a threshold of :math:`\alpha^2 = 75`,
to confirm that the effect is not by chance,
and show the parameters recovered when correcting for the bias (blue) and the
parameters recovered when you do not correct for the bias (red).


.. figure::     ../dessn/proofs/efficiency_5/output/surfaces.png
    :align:     center
    :width:     80%
"""