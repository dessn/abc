r""" In this model I want to test the applicability of efficiency to recovering unbiased
results from a biased set of data.

To this end, I implement a simple toy model, where we are observing points
from an underlying distribution, which is is normal and parameterised by :math:`\mu` and
:math:`\sigma`. To generate our data, we sample from this distribution, and denote
:math:`\vec{s}` the realised point of our distribution.

We then add in noise (our error), which takes the form of a normal distribution
centered around the realised value :math:`s`, with standard deviation :math:`e`,
where :math:`e` is some arbitrary number currently set to :math:`\mu/5`, where :math:`\mu` is
the true value, not the current value in the model.  Denote the observed data
points :math:`\vec{d}`.

Now, we bias our sample, by discarding all points which have a signal to noise
(:math:`d/e`) less than some threshold :math:`\alpha`.

To start from the beginning, we wish to calculate the posterior, and utilise Bayes theorem

.. math::
    P(\mu,\sigma|\vec{d},e) \propto P(\vec{d},e|\mu,\sigma) P(\mu,\sigma)

To simplify matters, we utilise flat priors and therefore neglect the :math:`P(\mu,\sigma)` term
as a constant multiplier.

.. math::
    P(\mu,\sigma|\vec{d},e) \propto P(\vec{d},e|\mu,\sigma)

    P(\mu,\sigma|\vec{d},e) \propto \int d\vec{s} P(\vec{d},e,\vec{s}|\mu,\sigma)

    P(\mu,\sigma|\vec{d},e) \propto \int d\vec{s} P(\vec{d},e|s,\mu,\sigma) P(s|\mu,\sigma)

where :math:`s` is the latent (true/underlying/actual) value of the point if we could remove
all observational error from :math:`\vec{d}`.  As :math:`\vec{s}` is our
realisation of the underlying distribution, which is normal, we can model its
conditional probability easily. Note in the analysis below, we have all observations
being independent of one another, and such the final probability is simply the product
of the probability for each observation.

.. math::
    P(\vec{s}|\mu,\sigma) = \prod_{i} \frac{1}{\sqrt{2\pi}\sigma}
    \exp\left[-\frac{(s_i-\mu)^2}{2\sigma^2}\right]

This leaves us with only one piece left in the puzzle, which is the probability we observe
a given value :math:`d_i` given the model parameters :math:`\lbrace s_i,\mu,\sigma \rbrace`.
It is important to note that the probability we observe value :math:`d` is, in our biased
observations, the probability that the value :math:`d` physically occurs, times by the
probability that it makes it into our final selection of data. We model this likelihood as

.. math::
    \mathcal{L} =  \prod_{i} \frac{\epsilon(d_i,e) P_g(d_i,e|s_i,\mu,\sigma)}
    {\int dR \ \epsilon(R,e) P_g(R,e|s_i,\mu,\sigma)}

where :math:`R` is used to denote a potential realisation of the data, given the underlying
model, and the subscript :math:`g` is again used to denote simple the data being generated,
but not necessarily also part of our sample (and thus different to the likelihood).

At this point, we should also note that :math:`P_g(d,e|s,\mu,\sigma) = P_g(d,e|s)`, as the
probability is independent of :math:`\mu` and :math:`\sigma`.

.. math::
    \mathcal{L} = \prod_{i} \frac{\epsilon(d_i,e) P_g(d_i,e|s_i)}
    {\int dR \ \epsilon(R,e) P_g(R,e|s_i)}


So, given a data point :math:`R` with observational error :math:`e`, and latent
parameter :math:`s_i`, we have that

.. math::
    \int dR \ \epsilon(R,e) P_g(R|s_i) = \int_{\alpha e}^{\infty}
    \frac{1}{\sqrt{2\pi}e} \exp\left[ -\frac{(R-s_i)^2}{2 e^2} \right] dR

Evaluating this by transforming coordinate to :math:`x = R-s_i` such that we get

.. math::
    \int dR \ \epsilon(R,e) P_g(R|s_i) = \int_{\alpha e - s_i}^{\infty}
    \frac{1}{\sqrt{2\pi}e} \exp\left[ -\frac{x^2}{2 e^2} \right] dx

gives the answer

.. math::
    \int dR \ \epsilon(R,e) P_g(R|s_i) = \begin{cases}
    \frac{1}{2} - {\rm erf} \left[ \frac{\alpha e - s_i}{\sqrt{2} e} \right] &
    \text{ if } \alpha e - s_i > 0 \\
    \frac{1}{2} + {\rm erf} \left[ \frac{s_i - \alpha e}{\sqrt{2} e} \right] &
    \text{ if } \alpha e - s_i < 0 \\
    \end{cases}


This gives us a final posterior that takes the form

.. math::
    P(\mu,\sigma|d,e) =  \int d\vec{s} \mathcal{L} P(\vec{s}|\mu,\sigma)


    P(\mu,\sigma|\vec{d},e) = \frac{1}{2\pi}
    \int d\vec{s} \frac{1}{e\sigma}\prod_{i} \frac{\exp\left[ -\frac{(d_i-s_i)^2}{2 e^2} -
    \frac{(s_i-\mu)^2}{2\sigma^2}\right]}
    {\begin{cases}
    \frac{1}{2} - {\rm erf} \left[ \frac{\alpha e - s_i}{\sqrt{2} e} \right] &
    \text{ if } \alpha e - s_i > 0 \\
    \frac{1}{2} + {\rm erf} \left[ \frac{s_i - \alpha e}{\sqrt{2} e} \right] &
    \text{ if } \alpha e - s_i < 0 \\
    \end{cases}}

I should also note in this implementation that I have neglected to treat :math:`d`
and :math:`s` properly, as they are both vectors and there should in fact be products in the
above equation (and the integral term should be over many different :math:`s_i`).


We can not implement this correction, and then implement it, and hopefully see that the
recovered underlying distribution becomes unbiased. This is not currently the case.

The model PGM:


.. figure::     ../dessn/models/test_efficiency_2/output/pgm.png
    :align:     center
    :width: 70%

The models, corrected and uncorrected. There appears to still be an issue that
we have not managed to resolve to get the result unbiased.

.. figure::     ../dessn/models/test_efficiency_2/output/surfaces.png
    :align:     center



"""