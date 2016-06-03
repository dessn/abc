r""" In this model I want to test the applicability of efficiency to recovering unbiased
results from a biased set of data.

This example follows from the previous one, but instead of having a single value :math:`\mu`
with data points and error around it, instead we have perfect realisations of an underlying
distribution parameterised by :math:`\mu` and :math:`\sigma`. This should be equivalent
to the previous example, except with one more dimension in the final output. More formally,

.. math::
    d\sim\mathcal{N}(\mu,\sigma).

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
in the previous example we define the likelihood :math:`\mathcal{L} = P(d|\mu,\sigma)` as

.. math::
    \mathcal{L} = P(D=d | D > \alpha, \mu, \sigma)

    \mathcal{L} = \frac{P(D=d, D > \alpha | \mu, \sigma)}
    {\int dR \ P(D > \alpha, D=R|\mu,\sigma)}

    \mathcal{L} = \frac{P(D > \alpha | D=d, \mu, \sigma) P(D=d|\mu,\sigma)}
    {\int dR \ P(D > \alpha|D=R,\mu,\sigma) P(D=R|\mu,\sigma)}

    \mathcal{L} = \frac{P(d > \alpha | d, \mu, \sigma) P(d|\mu,\sigma)}
    {\int dR \ P(R > \alpha|R,\mu,\sigma) P(R|\mu,\sigma)}

    \mathcal{L} = \frac{P(d > \alpha | d) P(d|\mu,\sigma)}
    {\int dR \ P(R > \alpha|R) P(R|\mu,\sigma)}

    \mathcal{L} = \frac{\epsilon(d,\alpha) P(d|\mu,\sigma)}
    {\int dR \ \epsilon(R,\alpha)  P(R|\mu,\sigma)}

where :math:`R` is used to denote a potential realisation of the data, given the underlying
model. The last line represents the general form of the likelihood which shall be presented
in the next examples. An efficiency, times by a probability, normalised over all potential
realisations of the data.

Putting in our efficiency and distributions

.. math::
    \mathcal{L} = \frac{\mathcal{H}(d-\alpha) \mathcal{N}(d;\mu,\sigma)}
    {\int dR \ \mathcal{H}(R-\alpha)  \mathcal{N}(R;\mu,\sigma)}

    \mathcal{L} = \frac{\mathcal{N}(d;\mu,\sigma)}
    {\int_\alpha^\infty dR \ \mathcal{N}(R;\mu,\sigma)}

Looking at the denominator:


.. math::
    \int_\alpha^\infty dR \ \mathcal{N}(R;\mu,\sigma) = \int_{\alpha}^{\infty}
    \frac{1}{\sqrt{2\pi}\sigma} \exp\left[ -\frac{(R-\mu)^2}{2 \sigma^2} \right] dR

Evaluating this by transforming coordinate to :math:`x = R-\mu` such that we get

.. math::
    \int_\alpha^\infty dR \ \mathcal{N}(R;\mu,\sigma) = \int_{\alpha - \mu}^{\infty}
    \frac{1}{\sqrt{2\pi}\sigma} \exp\left[ -\frac{x^2}{2 \sigma^2} \right] dx

gives the answer

.. math::
    \int_\alpha^\infty dR \ \mathcal{N}(R;\mu,\sigma) = g(\alpha, \mu, \sigma) = \begin{cases}
    \frac{1}{2} - \frac{1}{2}{\rm erf} \left[ \frac{\alpha - \mu}{\sqrt{2} \sigma} \right] &
    \text{ if } \alpha e - \mu > 0 \\
    \frac{1}{2} + \frac{1}{2}{\rm erf} \left[ \frac{\mu - \alpha}{\sqrt{2} \sigma} \right] &
    \text{ if } \alpha e - \mu < 0 \\
    \end{cases}

Which gives a final likelihood of

.. math::
    \mathcal{L} = \frac{\mathcal{N}(d;\mu,\sigma)}
    {g(\alpha, \mu,\sigma)}
We can not implement this correction, and then implement it, and hopefully see that the
recovered underlying distribution becomes unbiased. I plot three realisations of the data
to confirm that the effect is not by change.

The model PGM:

.. figure::     ../dessn/proofs/efficiency_2/output/pgm.png
    :width:     60%
    :align:     center

The posterior surfaces for both corrected (blue) and uncorrected (red) models.

.. figure::     ../dessn/proofs/efficiency_2/output/surfaces.png
    :align:     center
    :width:     80%

"""