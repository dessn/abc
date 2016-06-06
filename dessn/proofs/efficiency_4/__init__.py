r"""  In this efficiency proof, I again wish to increase my model complexity
and ensure unbiased recovery.

To help explain the model, I will willfully abuse certain terms, such as luminosity
and flux, to provide physical motivation for my parameterisation, even if I have
physically incorrect relationships.

Let us assume a supernova population, where a supernova luminosity is drawn
from an underlying population

.. math::
    L \sim \mathcal{N}(\mu, \sigma)

The supernova is located at some redshift :math:`z_o`, and, in this alternate reality
we are working in, the flux of the supernova is related as

.. math::
    f = \frac{L}{1 + z_o}

Finally, the observed flux has Poisson noise, giving

.. math::
    f_o \sim \mathcal{N}(f, \sqrt{f})

When we observe, we approximate the error :math:`\sqrt{f}` as :math:`\sqrt{f_o}`, and
discard points which have less a signal to noise less than :math:`\alpha`, which translates
into a selection cut at :math:`f_o > \alpha^2`.

Following previous examples, we assume flat priors, denote our latent (true) flux as
:math:`f`, arbitrary input flux as :math:`F`, and write out the likelihood as

.. math::
    \mathcal{L} = P(F = f_o, z_o|F > \alpha^2, \mu, \sigma)

    \mathcal{L} = \frac{P(F = f_o, z_o, F>\alpha^2 | \mu, \sigma)}
    {P(F>\alpha^2|\mu,\sigma)}

Now we enforce that the denominator is the normalisation constant over all possible
experimental outcomes (using a subscript R to denote realised vales). Note that
as the redshift is assumed to be perfectly determined, there is no integral
over redshift, the observed value represnts all possible experimental outcomes. Note that
in our model, we enforce a uniform redshift range from 0.5 to 1.5, to explain the integral bounds
that appear.

.. math::

    P(F>\alpha^2|\mu,\sigma) &= \int df_R \int dz_R \int dz \int dL \ P(F>\alpha^2, f_R, z_R, z, L |\mu,\sigma) \\
    &= \int df_R \int dz_R \int dz \int dL \ P(F>\alpha^2 | f_R) P(z_R | z) P(f_R | z, L) P(L |\mu,\sigma) P(z) \\
    &= \int df_R \int dz_R \int dL \ \mathcal{H}(f_R - \alpha^2) \mathcal{N}\left(f_R; \frac{L}{1+z_R}, \sqrt{\frac{L}{1+z_R}}\right) \mathcal{N}(L ;\mu,\sigma) \\
    &= \int_{-\infty}^{\infty} dL\ \mathcal{N}(L ;\mu,\sigma)\int_{0.5}^{1.5} dz_R  \int_{\alpha^2}^\infty df_R  \ \mathcal{N}\left(f_R; \frac{L}{1+z_R}, \sqrt{\frac{L}{1+z_R}}\right)  \\

Transforming from :math:`L` to :math:`f` via :math:`L = (1+z_R)f \rightarrow dL = df(1+z_R)`:

.. math::

    P(F>\alpha^2|\mu,\sigma) &= \int_{0.5}^{1.5} dz_R  \int_{-\infty}^{\infty} df \mathcal{N}(f(1+z_R) ;\mu,\sigma) \int_{\alpha^2}^\infty df_R  \ \mathcal{N}\left(f_R; f, \sqrt{f}\right)

Focusing on the last term and following the previous example, we show that

.. math::
    \int_{\alpha^2}^\infty df_R \ \mathcal{N}(f_R; f, \sqrt{f}) =
    \int_{\alpha^2}^\infty df_R \frac{1}{\sqrt{2\pi f}} \exp\left[-\frac{(f_R - f)^2}{2 f}\right]

We translate variables such that :math:`x = f_R - f`, which implies :math:`dx = df_R`, giving:

.. math::
     \int_{\alpha^2}^\infty df_R \ \mathcal{N}(f_R; f, \sqrt{f}) =
    \int_{\alpha^2-f}^\infty df_R \frac{1}{\sqrt{2\pi f}} \exp\left[-\frac{(x)^2}{2 f}\right]

Which evaluates to

.. math::
    \int_{\alpha^2}^\infty df_R \ \mathcal{N}(f_R; f, \sqrt{f}) = g(f, \alpha) = \begin{cases}
    \frac{1}{2} - \frac{1}{2}{\rm erf} \left[ \frac{\alpha^2 - f}{\sqrt{2f}} \right] &
    \text{ if } \alpha^2 - f > 0 \\
    \frac{1}{2} + \frac{1}{2}{\rm erf} \left[ \frac{f - \alpha^2}{\sqrt{2f}} \right] &
    \text{ if } \alpha^2 - f < 0 \\
    \end{cases}

Substituting this back in gives us a calculable denominator:

.. math::
    P(F>\alpha^2|\mu,\sigma) &= \int_{0.5}^{1.5} dz_R  \int_{-\infty}^{\infty} df\ \mathcal{N}(f(1+z_R) ;\mu,\sigma) g(f, \alpha)

Giving an update likelihood

.. math::
    \mathcal{L} &= \frac{P(F = f_o, z_o, F>\alpha^2 | \mu, \sigma)}{\int_{0.5}^{1.5} dz_R  \int_{-\infty}^{\infty} df\ \mathcal{N}(f(1+z_R) ;\mu,\sigma) g(f, \alpha)} \\
    \mathcal{L} &= \frac{(1+z_o) \int_{-\infty}^\infty df \ \mathcal{N}(f_o; f, \sqrt{f}) \mathcal{N}(f_o(1+z_o); \mu, \sigma)}{\int_{0.5}^{1.5} dz_R  \int_{-\infty}^{\infty} df\ \mathcal{N}(f(1+z_R) ;\mu,\sigma) g(f, \alpha)} \\

where in the last line we introduced the latent flux parameter in the numerator,
and converted form luminosity to flux, just as was done in the denominator.

We now can model this using generated data and test our model performance. The denominator term,
as a function of :math:`\mu`, :math:`\sigma` and :math:`z` can be thought of as a weighting, and
is shown below. Notice the similarity in the redshift slices to the weights shown in the
previous example.


.. figure::     ../dessn/proofs/efficiency_4/output/weights.png
    :width:     80%
    :align:     center

The model PGM is constructed as follows:

.. figure::     ../dessn/proofs/efficiency_4/output/pgm.png
    :width:     80%
    :align:     center

I plot three realisations of the data uaing a threshold of :math:`\alpha^2 = 75`,
to confirm that the effect is not by chance,
and show the parameters recovered when correcting for the bias (blue) and the
parameters recovered when you do not correct for the bias (red).


.. figure::     ../dessn/proofs/efficiency_4/output/surfaces.png
    :align:     center
    :width:     80%



"""