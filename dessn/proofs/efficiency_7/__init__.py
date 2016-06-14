r"""In this example we aim to show the ability of to use importance sampling on
datasets exhibiting non-significant bias to recover unbiased posteriors by including
efficiency weights on biased chains.

To show this, we revisit the 2D Efficiency example model, where data is effectively
realised from a truncated normal distribution.

.. math::
    d \sim \mathcal{N}(\mu, \sigma), \quad{\rm given}\quad d > \alpha

As shown in the maths for the the 2D Efficiency, our likelihood becomes

.. math::
    \mathcal{L} = \frac{\mathcal{N}(d;\mu,\sigma)}
    {g(\alpha, \mu,\sigma)}

where

.. math::
    g(\alpha, \mu, \sigma) &=  \int_\alpha^\infty dR \ \mathcal{N}(R;\mu,\sigma) \\
    &= \begin{cases}
    \frac{1}{2} - \frac{1}{2}{\rm erf} \left[ \frac{\alpha - \mu}{\sqrt{2} \sigma} \right] &
    \text{ if } \alpha - \mu > 0 \\
    \frac{1}{2} + \frac{1}{2}{\rm erf} \left[ \frac{\mu - \alpha}{\sqrt{2} \sigma} \right] &
    \text{ if } \alpha - \mu < 0 \\
    \end{cases}

It is important to note that this is for a single data point, and, given each
data point is independent, the probability for a sample of data points is the product
of the likelihoods for each individual point. As such, when invoking importance sampling on
the biased estimate using :math:`\mathcal{L} = \prod_{i}^N\mathcal{N}(d_i;\mu,\sigma)`,
we need to weight each sample by :math:`W = \left[g(\alpha, \mu, \sigma)\right]^{-N}`.


Shown below are the output surfaces generated when
setting :math:`\mu=100,\ \sigma=20,\ \alpha=50, \ N=2000`. In green is shown the model when
fit to all data, without the cut on alpha. Red is shown
the model when fit to the biased data, and as expected it produces biased results. Blue shows
the model fit to biased data, with each sample then reweighted according to :math:`W` defined
above.

.. figure::     ../dessn/proofs/efficiency_7/output/surfaces.png
    :align:     center
    :width:     80%


For those that wish to compare importance sampling against a model with implemented
bias correction (such as all previous models in these model proofs) the below figure shows
in blue the correct model (with bias correction applied inside the model likelihood) and
the orange contour shows importance sampling. We can see they produce essentially identical
surfaces, with only some slight deviation in the importance sampling due to sparsity of
samples at many sigma from the biased models point of maximum posterior. This could be fixed
by letting the chains run longer, to produce a more densely sampled parameter space.

.. figure::     ../dessn/proofs/efficiency_7/output/surfaces1.png
    :align:     center
    :width:     80%

We should note that importance sampling is only possible to use when the biased
in the output surfaces is small enough that the area of true high posterior is still
sufficiently sampled, and that for more significant cuts (and biased results) such
as those shown in the 2D efficiency example, one would have to implement the correction
in the model itself.

"""