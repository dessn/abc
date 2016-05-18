r""" This module is designed to give a step by step overview of a very simplified example
Bayesian model.

Setting up the math for some examples.

Let us assume that we are observing supernova that are drawn from an underlying
supernova distribution parameterised by :math:`\theta`,
where the supernova itself simply a luminosity :math:`L`. We measure the luminosity
of multiple supernovas, giving us an array of measurements :math:`D`. If we wish to recover
the underlying distribution of supernovas from our measurements, we wish to find :math:`P(\theta|D)`,
which is given by

.. math::
    P(\theta|D) \propto P(D|\theta)P(\theta)

Note that in the above equation, we realise that :math:`P(D|L) = \prod_{i=1}^N P(D_i|L_i)` as
our measurements are independent. The likelihood :math:`P(D|\theta)` is given by

.. math::
    P(D|\theta) =  \prod_{i=1}^N  \int_{-\infty}^\infty P(D_i|L_i) P(L_i|\theta) dL_i



We now have two distributions to characterise. Let us assume both are gaussian, that is
our observed luminosity :math:`x_i` has gaussian error :math:`\sigma_i` from the actual supernova
luminosity, and the supernova luminosity is drawn from an underlying gaussian distribution
parameterised by :math:`\theta`.

 .. math::
    P(D_i|L_i) = \frac{1}{\sqrt{2\pi}\sigma_i}\exp\left(-\frac{(x_i-L_i)^2}{2\sigma_i^2}\right)

    P(L_i|\theta) = \frac{1}{\sqrt{2\pi}\theta_2}\exp\left(-\frac{(L_i-\theta_1)^2}{2\theta_2^2}\right)



This gives us a likelihood of

.. math::

    P(D|\theta) = \prod_{i=1}^N  \frac{1}{2\pi \theta_2 \sigma_i}  \int_{-\infty}^\infty
    \exp\left(-\frac{(x_i-L_i)^2}{2\sigma_i^2} -\frac{(L_i-\theta_1)^2}{2\theta_2^2} \right) dL_i


Working in log space for as much as possible will assist in numerical precision, so we can
rewrite this as

.. math::
    \log\left(P(D|\theta)\right) =  \sum_{i=1}^N  \left[
            \log\left( \int_{-\infty}^\infty \exp\left(-\frac{(x_i-L_i)^2}{2\sigma_i^2} -
    \frac{(L_i-\theta_1)^2}{2\theta_2^2} \right) dL_i \right) -\log(2\pi\theta_2\sigma_i) \right]


"""