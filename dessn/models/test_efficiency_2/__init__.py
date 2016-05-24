r""" In this model I want to test the applicability of efficiency to recovering unbiased
results from a biased set of data.

This example follows from the previous one, but instead of having a single value :math:`\mu`
with data points and error around it, instead we have perfect realisations of an underlying
distribution parameterised by :math:`\mu` and :math:`\sigma`. This should be equivalent
to the previous example, except with one more dimension.


Now, we bias our sample, by discarding all points which have an observed value :math:`d`
less than some threshold :math:`\alpha`.

Like normal we seek to construct the posterior

.. math::
    P(\mu,\sigma|d) \propto P(d|\mu,\sigma) P(\mu,\sigma)

Considering a flat prior on :math:`P(\mu,\sigma)` and dropping the term as a constant
multiplier, we now have to consider the likelihood of observing our data :math:`d`
given our model :math:`\mu`. Given we bias our data, this becomes the probability of
the data being generated at :math:`d` given :math:`\mu`, multiplied by the probability
that the data :math:`d` *also* makes it into our final dataset and is not dropped. As
such we define the likelihood :math:`\mathcal{L} = P(d|\mu,\sigma)` as

.. math::
    \mathcal{L} = \frac{\epsilon(d) P_g(d|\mu,\sigma)}
    {\int dR \ \epsilon(R) P_g(R|\mu,\sigma)}

where :math:`R` is used to denote a potential realisation of the data, given the underlying
model and the subscript :math:`g` is used to denote "generated".


.. math::
    \mathcal{L} = \frac{\epsilon(d) P_g(d|\mu,\sigma)}{\int dR \ \epsilon(R) P_g(R|\mu,\sigma) }

So, given a data point :math:`R` with underlying distribution width :math:`\sigma`, and
underlying mean :math:`\mu`, we have that

.. math::
    \int dR \ \epsilon(R) P_g(R|\mu,\sigma) = \int_{\alpha}^{\infty}
    \frac{1}{\sqrt{2\pi}\sigma} \exp\left[ -\frac{(R-\mu)^2}{2 \sigma^2} \right] dR

Evaluating this by transforming coordinate to :math:`x = R-\mu` such that we get

.. math::
    \int dR \ \epsilon(R) P_g(R|\mu,\sigma) = \int_{\alpha - \mu}^{\infty}
    \frac{1}{\sqrt{2\pi}\sigma} \exp\left[ -\frac{x^2}{2 \sigma^2} \right] dx

gives the answer

.. math::
    \int dR \ \epsilon(R) P_g(R|\mu,\sigma) = \frac{1}{2} {\rm erfc}
    \left[ \frac{\alpha - \mu}{\sqrt{2} \sigma} \right]


We can not implement this correction, and then implement it, and hopefully see that the
recovered underlying distribution becomes unbiased. I plot three realisations of the data
to confirm that the effect is not by change.

The model PGM:

.. figure::     ../dessn/models/test_efficiency_2/output/pgm.png
    :width:     60%
    :align:     center

The posterior surfaces for both corrected (blue) and uncorrected (red) models.

.. figure::     ../dessn/models/test_efficiency_2/output/surfaces.png
    :align:     center

"""