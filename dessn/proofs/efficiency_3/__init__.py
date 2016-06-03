r""" In this model I want to test the applicability of efficiency to recovering unbiased
results from a biased set of data.

To this end, I implement a simple toy model, where we are observing points
from an underlying distribution, which is is normal and parameterised by :math:`\mu` and
:math:`\sigma`. To generate our data, we sample from this distribution, and denote
:math:`s` the realised point of our distribution.

We then add in noise (our error), which takes the form of a normal distribution
centered around the realised value :math:`s`, with standard deviation :math:`e`,
where :math:`e` is some arbitrary number currently set to :math:`\mu/5`, where :math:`\mu` is
the true value, not the current value in the model.  Denote the observed data
points :math:`d`. Formally

.. math::
    s \sim \mathcal{N}(\mu, \sigma)

    d \sim \mathcal{N}(s,e)

Now, we bias our sample, by discarding all points which have a signal to noise
(:math:`d/e`) less than some threshold :math:`\alpha`.

To start from the beginning, we wish to calculate the posterior, and utilise Bayes theorem

.. math::
    P(\mu,\sigma,e|d) \propto P(d|\mu,\sigma,e) P(\mu,\sigma,e)

To simplify matters, we utilise flat priors and therefore neglect the :math:`P(\mu,\sigma,e)` term
as a constant multiplier.

.. math::
    P(\mu,\sigma,e|d) \propto P(d|\mu,\sigma,e)

As the detailed maths has been shown in examples 1 and 2, here I simply write the general
expression for a biased likelihood:

.. math::
    \mathcal{L} = \frac{\epsilon(d,\alpha, e) P(d|\mu,\sigma, e)}
    {\int dR \ \epsilon(R,\alpha, e)  P(R|mu,\sigma, e)}

To model the hidden layer containing :math:`s`, we introduce it in both the denominator and
the numerator.

.. math::
    \mathcal{L} = \frac{\int ds \ \epsilon(d,\alpha, e) P(d, s|\mu,\sigma, e)}
    {\int dR \int ds\ \epsilon(R,\alpha, e)  P(R, s|\mu,\sigma, e)}

.. math::
    \mathcal{L} = \frac{\int ds \ P(d|s,e) P(s|\mu,\sigma)}
    {\int dR \int ds\ \epsilon(R,\alpha, e)  P(R|s, e) P(s|\mu,\sigma)}

.. math::
    \mathcal{L} = \frac{\int ds \ \mathcal{N}(d;s,e) \mathcal{N}(s;\mu,\sigma)}
    {\int ds\ \mathcal{N}(s;\mu,\sigma) \int dR \ \mathcal{H}(R - \alpha e)  \mathcal{N}(R;s, e) }

Looking at the denominator:

.. math::
    \int dR \ \mathcal{H}(R - \alpha e)  \mathcal{N}(R;s, e)
    = \int_{\alpha e}^\infty \frac{1}{\sqrt{2\pi}e}
    \exp\left[ -\frac{(R - s)^2}{2e^2} \right] dR

Evaluating this by transforming coordinate to :math:`x = R-s` such that we get

.. math::
    \int dR \ \mathcal{H}(R - \alpha e)  \mathcal{N}(R;s, e)
    = \int_{\alpha e - s}^{\infty}
    \frac{1}{\sqrt{2\pi}e} \exp\left[ -\frac{x^2}{2 e^2} \right] dx

gives the answer

.. math::
    \int dR \ \mathcal{H}(R - \alpha e)  \mathcal{N}(R;s, e) = g(s, \alpha, e) = \begin{cases}
    \frac{1}{2} - \frac{1}{2}{\rm erf} \left[ \frac{\alpha e - s}{\sqrt{2} e} \right] &
    \text{ if } \alpha e - s > 0 \\
    \frac{1}{2} + \frac{1}{2}{\rm erf} \left[ \frac{s - \alpha e}{\sqrt{2} e} \right] &
    \text{ if } \alpha e - s < 0 \\
    \end{cases}

Which gives a likelihood of

.. math::
    \mathcal{L} = \frac{\int ds \ \mathcal{N}(d;s,e) \mathcal{N}(s;\mu,\sigma)}
    {\int ds\ \mathcal{N}(s;\mu,\sigma) g(s, \alpha, e)}

In terms of implementation details, as :math:`\alpha` and :math:`e` are fixed
at the start of the model, we can precompute the normalisation term. And as the normalisation
term is dependent in our example on :math:`\mu` and :math:`\sigma`, we can easily
plot this surface, which has been done below when translated into a probabalistic
weighting. In English, the probability on the surface is the probability that
we *would* observe data for the given model parametrisation. Notice how, having
set :math:`\alpha=3.25` and :math:`e=20` in the generated data, we observe the
symmetry axis at :math:`\alpha e = 65`. If :math:`\mu` is above this value, we are
more likely to observe data the smaller :math:`\sigma`, and below the value
we are more likely to observe given higher :math:`\sigma`.

.. figure::     ../dessn/proofs/efficiency_3/output/weights.png
    :align:     center
    :width:     80%


We can not implement this correction, and then implement it, and hopefully see that the
recovered underlying distribution becomes unbiased.

The model PGM:


.. figure::     ../dessn/proofs/efficiency_3/output/pgm.png
    :align:     center
    :width:     80%

The models, corrected (blue), uncorrected (red) and if the data was not biased
(green). We show two random realisations of the data.

.. figure::     ../dessn/proofs/efficiency_3/output/surfaces.png
    :align:     center
    :width:     80%



"""