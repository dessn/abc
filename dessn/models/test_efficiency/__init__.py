r""" In this model I want to test the applicability of efficiency to recovering unbiased
results from a biased set of data.

To this end, I implement a simple toy model, where we are observing points
from an underlying distribution, which is is normal and parameterised by :math:`\mu` and
:math:`\sigma`. To generate our data, we sample from this distribution,
and then add in noise (our error), which takes the form of a normal distribution
centered around the actual observed value, with standard deviation :math:`\sqrt{s}`,
where :math:`s` is the realised point of our distribution. Denote the observed data point
as :math:`d`.

Given our observed data point value, we give it some observational error (corresponding
to a Poisson process), such that the error on :math:`d` is given by :math:`e \equiv \sqrt{d}`.

Now, we bias our sample, by discarding all points which have a signal to noise
(:math:`d/\sqrt{d}=\sqrt{d}`) less than some threshold :math:`\alpha`.

We model the likelihood as

.. math::
    \mathcal{L} = \frac{\epsilon(D|\theta) P(D|\theta)}{\int dR \epsilon(R|\theta) P(R|\theta)}

where :math:`R` is used to denote a potential realisation of the data, given the underlying
model :math:`\theta` (which includes the latent points, and underlying distribution mean
and sigma).

So, given a data point :\math:`R` with observational error :math:`\sqrt{R}`, and latent
parameter :math:`p`, we have that

.. math::
    \int dR \epsilon(R|\theta) P(R|\theta) = \int_{\alpha \sigma}^{\infty}
    \frac{1}{\sqrt{2\pi R}} \exp\left[ -\frac{(R-p)^2}{2 R^2} \right]

Evaluating this gives the answer

.. math::
    \int dR \epsilon(R|\theta) P(R|\theta) = \frac{1}{2} {\rm erfc}
    \left[ \frac{\alpha \sqrt{R} - p}{\sqrt{2 R} } \right]


We can not implement this correction, and then implement it, and hopefully see that the
recovered underlying distribution becomes unbiased. This is not currently the case.

The uncorrected model:

.. figure::     ../dessn/models/test_efficiency/output/surface_no.png
    :align:     center

The corrected model (so obviously the correction is incorrect):

.. figure::     ../dessn/models/test_efficiency/output/surface_cor.png
    :align:     center



"""