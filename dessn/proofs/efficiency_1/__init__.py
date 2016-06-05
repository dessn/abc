r""" In this model I want to test the applicability of efficiency to recovering unbiased
results from a biased set of data.

To this end, I implement the simplest toy model, where we are observing points scattered
from their true value by observational error. The true underlying value is given by :math:`\mu`,
and given an error :math:`e`, we have draw our observed value from

.. math::
    d \sim \mathcal{N}(\mu, e)

For notational simplicity, I will be treating :math:`d` as a single value and not
the actual vector that it is. However, as we have independent data, this is not
an issue for us.

Now, we bias our sample, by discarding all points which have a signal to noise
(:math:`d/e`) less than some threshold :math:`\alpha`. Phrased differently,
all our data samples must now satisfy the condition that :math:`D - \alpha e > 0`. Note
that we denote a data sample here as :math:`D`, which is distinct to the experimental
data :math:`d`. :math:`D` is simply our model stating that *any* input data must
satisfy the selection criterion.

Like normal we seek to construct the posterior

.. math::
    P(\mu|d) \propto P(D = d|D > \alpha e, \mu) P(\mu)

Considering a flat prior on :math:`P(\mu)` and dropping the term as a constant
multiplier, and reducing the expression simply to a likelihood

.. math::
    \mathcal{L} = P(D = d|D > \alpha e, \mu)

In English, this can be read as: *"Given the current model parameters, and our
stipulation that the input data satisfy our selection criterion, what is
the probability of realising the experimental data"*.

Now, performing some mathematics, we can get this to a better state:

.. math::
    \mathcal{L} = P(D = d|D > \alpha e, \mu)

    \mathcal{L} = \frac{P(D=d, D>\alpha e | \mu)}{P(D > \alpha e | \mu)}

It is important to note here that, by itself, :math:`P(D > \alpha e | \mu)` is not
quantifiable until :math:`D` is set to some realisation of the data. In other
words, :math:`D` represents *"the input data"*, and thus requires input data to
evaluate. We do this by introducing an integral over all data realisations. Note that
this is identical to enforcing the normalisation of the likelihood over all data (which
must be true).

.. math::
    \mathcal{L} = \frac{P(D=d, D>\alpha e | \mu) }{\int dR\ P(D > \alpha e, D=R | \mu)}

    \mathcal{L} = \frac{P(d, d>\alpha e | \mu) }{\int dR\ P(R > \alpha e, R | \mu)}

    \mathcal{L} = \frac{P(d>\alpha e | d \mu) P(d|\mu)}{\int dR\ P(R > \alpha e, R | \mu) P(R|\mu)}

We call the term :math:`P(d>\alpha e | d \mu)` the efficiency, for it is the probability
that, given input data and a model parametrisation, our data passes our cuts. We denote this
function :math:`\epsilon(d)`, where we should note that the other arguments on which the
efficiency depends (:math:`\mu`, :math:`\alpha` and :math:`e`) are not written
for convenience as they are either held constant (:math:`\alpha` and :math:`e`)
or :math:`\epsilon` is independent of them (in the case of :math:`\mu`).

.. math::
    \mathcal{L} = \frac{\epsilon(d) P(d|\mu)}{\int dR\ \epsilon(R) P(R|\mu)}

As stated previously, we model :math:`P(d|\mu)` with distribution :math:`\mathcal{N}(\mu, e)`.
The efficiency of all actually observed data must be 1 (as it *did* pass our cuts), so
:math:`\epsilon(d) = 1`. The efficiency of all realisations is more complicated, and
it takes the form of a Heaviside step function :math:`\mathcal{H}(R - \alpha e)`. Putting
this together:

.. math::
    \mathcal{L} = \frac{\mathcal{N}(d;\mu, e)}
    {\int_{-\infty}^\infty dR\ \mathcal{H}(\alpha e) \mathcal{N}(R;\mu, e)}

Using the Heaviside step function to modify the integral limits, we see that the
denominator becomes a truncated cumulative normal distribution function.

.. math::
    \mathcal{L} = \frac{\mathcal{N}(d;\mu,e)}{\int_{\alpha e}^\infty dR\ \mathcal{N}(R;\mu,e)}

Looking only at the denominator:

.. math::
    \int_{\alpha e}^\infty dR\ \mathcal{N}(R;\mu,e) =
    \int_{\alpha e}^{\infty}\frac{1}{\sqrt{2\pi}e} \exp\left[ -\frac{(R-\mu)^2}{2 e^2} \right] dR


Evaluating this by transforming coordinate to :math:`x = R-\mu` such that we get

.. math::
    \int dR \ \epsilon(R,e) P_g(R|\mu) = \int_{\alpha e - \mu}^{\infty}
    \frac{1}{\sqrt{2\pi}e} \exp\left[ -\frac{x^2}{2 e^2} \right] dx

gives the answer

.. math::
     \int_{\alpha e}^{\infty} dR\ \mathcal{N}(R;\mu,e) = \begin{cases}
    \frac{1}{2} - \frac{1}{2}{\rm erf} \left[ \frac{\alpha e - \mu}{\sqrt{2} e} \right] &
    \text{ if } \alpha e - \mu > 0 \\
    \frac{1}{2} + \frac{1}{2}{\rm erf} \left[ \frac{\mu- \alpha e}{\sqrt{2} e} \right] &
    \text{ if } \alpha e - \mu < 0 \\
    \end{cases}

Restricting ourselves to :math:`\mu > \alpha e`, we have

.. math::
     \int_{\alpha e}^{\infty} dR\ \mathcal{N}(R;\mu,e) =
     \frac{1}{2} + \frac{1}{2}{\rm erf} \left[ \frac{\mu- \alpha e}{\sqrt{2} e} \right]

     \int_{\alpha e}^{\infty} dR\ \mathcal{N}(R;\mu,e) =
     \frac{1}{2}{\rm erfc} \left[ \frac{\alpha e - \mu}{\sqrt{2} e} \right]


Giving a final expression for our likelihood (and posterior in this case) of


.. math::
    P(\mu|d) \propto \frac{\mathcal{N}(d;\mu,e)}
    {\frac{1}{2}{\rm erfc} \left[ \frac{\alpha e - \mu}{\sqrt{2} e} \right]}

We can fit biased data (data which has gone through the selection cut) with and without
the :math:`{\rm erfc}` correction, to simulate respectively if we model the bias or
do not model the bias.


The model PGM:


.. figure::     ../dessn/proofs/efficiency_1/output/pgm.png
    :width:     60%
    :align:     center

I plot three realisations of the data to confirm that the effect is not by chance,
and show the parameters recovered when correcting for the bias (blue) and the
parameters recovered when you do not correct for the bias (red).


.. figure::     ../dessn/proofs/efficiency_1/output/surfaces.png
    :align:     center
    :width:     80%



"""