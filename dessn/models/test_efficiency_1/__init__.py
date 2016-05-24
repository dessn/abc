r""" In this model I want to test the applicability of efficiency to recovering unbiased
results from a biased set of data.

To this end, I implement the simplest toy model, where we are observing points scattered
from their true value by observational error. The true underlying value is given by :math:`\mu`.


We generate observed data points by adding in noise (our error), which takes the form
of a normal distribution centered around the true value :math:`\mu` with standard
deviation :math:`e`, where :math:`e` is some arbitrary number currently set to
:math:`\mu/5`, where :math:`\mu` is the true value, not the current value in the model.
Denote the observed data point :math:`d`, and consider the reported error :math:`e`
an observed value as well. For notational simplicity, I will be treating :math:`d` as a
single value and not the actual vector that it is. For more rigorous vector treatment,
see the next example.

Now, we bias our sample, by discarding all points which have a signal to noise
(:math:`d/e`) less than some threshold :math:`\alpha`.

Like normal we seek to construct the posterior

.. math::
    P(\mu|d,e) \propto P(d,e|\mu) P(\mu)

Considering a flat prior on :math:`P(\mu)` and dropping the term as a constant
multiplier, we now have to consider the likelihood of observing our data :math:`d`
given our model :math:`\mu`. Given we bias our data, this becomes the probability of
the data being generated at :math:`d` given :math:`\mu`, multiplied by the probability
that the data :math:`d` *also* makes it into our final dataset and is not dropped. As
such we define the likelihood :math:`\mathcal{L} = P(d,e|\mu)` as

.. math::
    \mathcal{L} = \frac{\epsilon(d,e) P_g(d,e|\mu)}
    {\int dR \ \epsilon(R,e) P_g(R,e|\mu)}

where :math:`R` is used to denote a potential realisation of the data, given the underlying
model and the subscript :math:`g` is used to denote "generated".


.. math::
    \mathcal{L} = \frac{\epsilon(d,e) P_g(d,e|\mu)}{\int dR \ \epsilon(R,e) P_g(R,e|\mu) }

So, given a data point :math:`R` with observational error :math:`e`, and
underlying mean :math:`\mu`, we have that

.. math::
    \int dR \ \epsilon(R,e) P_g(R|\mu) = \int_{\alpha e}^{\infty}
    \frac{1}{\sqrt{2\pi}e} \exp\left[ -\frac{(R-\mu)^2}{2 e^2} \right] dR

Evaluating this by transforming coordinate to :math:`x = R-\mu` such that we get

.. math::
    \int dR \ \epsilon(R,e) P_g(R|\mu) = \int_{\alpha e - \mu}^{\infty}
    \frac{1}{\sqrt{2\pi}e} \exp\left[ -\frac{x^2}{2 e^2} \right] dx

gives the answer

.. math::
    \int dR \ \epsilon(R,e) P_g(R|\mu) = \frac{1}{2} {\rm erfc}
    \left[ \frac{\alpha e - \mu}{\sqrt{2} e} \right]


We can not implement this correction, and then implement it, and hopefully see that the
recovered underlying distribution becomes unbiased. I plot three realisations of the data
to confirm that the effect is not by change.

The model PGM:


.. figure::     ../dessn/models/test_efficiency_1/output/pgm.png
    :width:     60%
    :align:     center

The posterior surfaces for both corrected (blue) and uncorrected (red) models.

.. figure::     ../dessn/models/test_efficiency_1/output/surfaces.png
    :align:     center



"""