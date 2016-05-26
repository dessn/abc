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

Now here we again need to differentiate between the probability that certain data
is generated from our model and the probability that the data is generated and
*successfully* observed. We denote the probability that the data is
generated (unobserved) with :math:`P_g(\vec{d},e|\mu,\sigma)`, and the
probability that the data is both generated and observed to be :math:`P(\vec{d},e|\mu,\sigma)`.
We denote the probability of the data, if generated, being observed using an
efficiency :math:`\epsilon(\vec{d},e)` (where we assume this probability is independent
of :math:`\mu` and :math:`\sigma`). Given we want our likelihood to be normalised over all
possible observed realisations of our data to be unity, we have that

.. math::
    \mathcal{L} = P(\vec{d},e|\mu,\sigma) = \frac{ \epsilon(\vec{d},e) P_g(\vec{d},e|\mu,\sigma)}
    {\int dR\  \epsilon(\vec{R},e) P_g(\vec{R},e|\mu,\sigma)}

where :math:`\vec{R}` represents all possible realisations of our data. Now,
we introduce the latent parameter vector :math:`\vec{s}`, representing the
realised points in the distribution parameterised by :math:`\mu` and :math:`\sigma`.
This differs to :math:`\vec{d}`, as :math:`\vec{d}` contains our observational noise.

.. math::
    \mathcal{L} = \frac{\int d\vec{s} \ \epsilon(\vec{d},e) P_g(\vec{d},e,\vec{s}|\mu,\sigma)}
    {\int dR \ \int d\vec{s}\  \epsilon(\vec{R},e) P_g(\vec{R},e,\vec{s}|\mu,\sigma)}

    \mathcal{L} = \frac{\int d\vec{s} \ \epsilon(\vec{d},e) P_g(\vec{d},e|\vec{s}) P(\vec{s}|\mu,\sigma)}
    {\int dR \ \int d\vec{s}\  \epsilon(\vec{R},e) P_g(\vec{R},e|\vec{s}) P(\vec{s}|\mu,\sigma)}

    \mathcal{L} = \frac{\int d\vec{s} \ \epsilon(\vec{d},e) P_g(\vec{d},e|\vec{s}) P(\vec{s}|\mu,\sigma)}
    {\int d\vec{s} \ P(\vec{s}|\mu,\sigma) \  \int d\vec{R}\ \epsilon(\vec{R},e) P_g(\vec{R},e|\vec{s}) }

Before continuing further, let us realise that each data point in :math:`\vec{d}` and
each realised point :math:`\vec{s}` are independent of one another, and so we can write out
that

.. math::
    \mathcal{L} = \prod_i \frac{\int d s_i \ \epsilon(d_i,e) P_g(d_i,e|s_i) P(s_i|\mu,\sigma)}
    {\int ds_i \ P(s_i|\mu,\sigma) \  \int dR_i\ \epsilon(R_i,e) P_g(R_i,e|s_i) }

On a different note, if we impose a :math:`d_i/e < \alpha` cut in our data, this gives us
observational bounds from :math:`d_i = \alpha e \rightarrow \infty`.
Considering the integral over potential realisations of observed data

.. math::
    \int dR_i\ \epsilon(R_i,e) P_g(R_i,e|s_i) = \int_{\alpha e}^\infty \frac{1}{\sqrt{2\pi}e}
    \exp\left[ -\frac{(R_i - s_i)^2}{2e^2} \right]

Evaluating this by transforming coordinate to :math:`x = R_i-s_i` such that we get

.. math::
    \int dR_i \ \epsilon(R_i,e) P_g(R_i, e|s_i) = \int_{\alpha e - s_i}^{\infty}
    \frac{1}{\sqrt{2\pi}e} \exp\left[ -\frac{x^2}{2 e^2} \right] dx

gives the answer

.. math::
    \int dR_i \ \epsilon(R_i,e) P_g(R_i,e|s_i) = f(s_i, \alpha, \epsilon) = \begin{cases}
    \frac{1}{2} - \frac{1}{2}{\rm erf} \left[ \frac{\alpha e - s_i}{\sqrt{2} e} \right] &
    \text{ if } \alpha e - s_i > 0 \\
    \frac{1}{2} + \frac{1}{2}{\rm erf} \left[ \frac{s_i - \alpha e}{\sqrt{2} e} \right] &
    \text{ if } \alpha e - s_i < 0 \\
    \end{cases}

Again looking at a different component, let us examine :math:`P(s_i|\mu,\sigma)`.

.. math::
    P(s_i|\mu,\sigma) = \frac{1}{\sqrt{2\pi}\sigma}
    \exp\left[-\frac{(s_i-\mu)^2}{2\sigma^2}\right]

Finally, we note that, if the data point is already in our sample, the
efficiency must be one.

Combined, this gives a likelihood (which is proportional to the posterior)
of

.. math::
    \mathcal{L} = \prod_i \frac{\int d s_i \ \mathcal{N}(d_i - s_i, e) \mathcal{N}(s_i - \mu, \sigma)}
    {\int ds \ \mathcal{N}(s - \mu, \sigma) \  f(s, \alpha, \epsilon) }

In terms of implementation details, as :math:`\alpha` and :math:`\epsilon` are fixed
at the start of the model, we can precompute

We can not implement this correction, and then implement it, and hopefully see that the
recovered underlying distribution becomes unbiased.

The model PGM:


.. figure::     ../dessn/models/test_efficiency_3/output/pgm.png
    :align:     center
    :width:     70%

The models, corrected and uncorrected. There appears to still be an issue that
we have not managed to resolve to get the result unbiased.

.. figure::     ../dessn/models/test_efficiency_3/output/surfaces.png
    :align:     center



"""