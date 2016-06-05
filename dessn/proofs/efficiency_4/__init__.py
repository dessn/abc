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
over redshift, the observed value represnts all possible experimental outcomes.

.. math::
    \mathcal{L} = \frac{P(F = f_o, z_o, F>\alpha^2 | \mu, \sigma)}
    {\int df_R \ P(F>\alpha^2, F=f_R, z_o|\mu,\sigma)}

Continuing with the algebra, we note that the luminosity is a function of flux
and redshift.

.. math::
    \mathcal{L} = \frac{P(f_o, z_o, f_o>\alpha^2 | \mu, \sigma)}
    {\int df_R \ P(f_R>\alpha^2, f_R, z_o|\mu,\sigma)}

    \mathcal{L} = \frac{P(f_o>\alpha^2 |f_o, z_o, \mu, \sigma) P(f_o, z_o | \mu, \sigma)}
    {\int df_R\ P(f>\alpha^2|f, z_o,\mu,\sigma) P(f, z_o | \mu, \sigma)}

Noting if we observed :math:`f_o`, is must have passed out cuts and thus be :math:`f_o > \alpha^2`.

.. math::
    \mathcal{L} = \frac{P(f_o, z_o | \mu, \sigma)}
    {\int df_R \ P(f_R>\alpha^2|f_R) P(f_R, z_o| \mu, \sigma)}

Adding in the latent parameters to represent the actual flux:

.. math::
    \mathcal{L} = \frac{\int df \ P(f_o, z_o, f| \mu, \sigma)}
    {\int df_R \ P(f_R>\alpha^2|f_R) \int df \ P(f_R, z_o, f| \mu, \sigma)}

Writing in the bounds now for our integrals (instead of the two step process of writing
the Heaviside step function and then using that to modify the integral limits), and remembering
that the observed flux is drawn from a normal distribution centered on the actual flux:

.. math::
    \mathcal{L} = \frac{\int_0^\infty df \ P(f_o, z_o, f| \mu, \sigma)}
    {\int_0^\infty df_R\ P(f_R>\alpha^2|f_R) \int_0^\infty df \ P(f_R, z_o, f| \mu, \sigma)}

    \mathcal{L} = \frac{\int_0^\infty df \ P(f_o, |z_o, f, \mu, \sigma) P(z_o, f| \mu, \sigma)}
    {\int_0^\infty df_R\ P(f_R>\alpha^2|f_R) \int_0^\infty df \ P(f_R|z_R, f, \mu, \sigma) P(z_o, f| \mu, \sigma)}

    \mathcal{L} = \frac{\int_0^\infty df \ P(f_o, |f) P(z_o, f| \mu, \sigma)}
    {\int_0^\infty df_ \ P(f_R>\alpha^2|f_R) \int_0^\infty df \ P(f_R|f) P(z_o, f| \mu, \sigma)}

    \mathcal{L} = \frac{\int_0^\infty df \ \mathcal{N}(f_o; f, \sqrt{f}) P(L_o| \mu, \sigma)}
    {\int_{\alpha^2}^\infty df_R \int_0^\infty df \ \mathcal{N}(f_R; f, \sqrt{f}) P(L| \mu, \sigma)}

    \mathcal{L} = \frac{\int_0^\infty df \ \mathcal{N}(f_o; f, \sqrt{f}) \mathcal{N}(L; \mu, \sigma)}
    {\int_{\alpha^2}^\infty df_R  \int_0^\infty df \ \mathcal{N}(f_R; f, \sqrt{f}) \mathcal{N}(L_R; \mu, \sigma)}

    \mathcal{L} = \frac{\int_0^\infty df \ \mathcal{N}(f_o; f, \sqrt{f}) \mathcal{N}(L; \mu, \sigma)}
    {\int_0^\infty df \ \mathcal{N}(L_R; \mu, \sigma) \int_{\alpha^2}^\infty df_R \ \mathcal{N}(f_R; f, \sqrt{f})}

Following the previous example, we show that

.. math::
    \int_{\alpha^2}^\infty df_R \ \mathcal{N}(f_R; f, \sqrt{f}) =
    \int_{\alpha^2}^\infty df_R \frac{1}{\sqrt{2\pi f}} \exp\left[-\frac{(f_R - f)^2}{2 f}\right]

We translate variables such that :math:`x = f_R - f`, which implies :math:`dx = df_R`, giving:

.. math::
     \int_{\alpha^2}^\infty df_R \ \mathcal{N}(f_R; f, \sqrt{f}) =
    \int_{\alpha^2-f}^\infty df_R \frac{1}{\sqrt{2\pi f}} \exp\left[-\frac{(x)^2}{2 f}\right]

Which evaluates to

.. math::
    \int_{\alpha^2}^\infty df_R \ \mathcal{N}(f_R; f, \sqrt{f}) = g(f) = \begin{cases}
    \frac{1}{2} - \frac{1}{2}{\rm erf} \left[ \frac{\alpha^2 - f}{\sqrt{2f}} \right] &
    \text{ if } \alpha^2 - f > 0 \\
    \frac{1}{2} + \frac{1}{2}{\rm erf} \left[ \frac{f - \alpha^2}{\sqrt{2f}} \right] &
    \text{ if } \alpha^2 - f < 0 \\
    \end{cases}

Placing this expression back into the likelihood gives us

.. math::
    \mathcal{L} = \frac{\int_0^\infty df \ \mathcal{N}(f_o; f, \sqrt{f}) \mathcal{N}(L; \mu, \sigma)}
    {\int_0^\infty df \ \mathcal{N}(L; \mu, \sigma) g(f)}

Finally, noting that :math:`L = f(1+z) \rightarrow \frac{dL}{df} = (1+z) \rightarrow dL = (1+z) df`,
we can write the normal distributions in luminosity as a function of flux.

.. math::
    \mathcal{L} = \frac{\int_0^\infty df \ \mathcal{N}(f_o; f, \sqrt{f}) (1+z_o)\mathcal{N}(f(1+z_o); \mu, \sigma)}
    {\int_0^\infty df \  (1+z_o)\mathcal{N}(f(1+z_o); \mu, \sigma) g(f)}

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