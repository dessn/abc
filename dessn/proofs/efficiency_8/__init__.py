r"""Having shown that importance sampling works for our general issue, we
will employ that methodology in this example.

We want to build on the :ref:`efficiency6` example, which introduced bands and
zero point calibrations, and we now make have time dependent luminosity.

We now model the luminosity as

.. math::
    L = L_0 \exp\left[ - \frac{(t-t_0)^2}{2s_0^2} \right],

where :math:`L_0` represents the maximum luminosity, :math:`t_0` the time
of maximum luminosity and :math:`s_0` is the stretch parameter. We assume that
the probability of maximum time remains constant, and seek to characterise
the distribution of luminosity and stretch, which are modelled as normal distrubtions.
We then also implement the same selection criteria as before - at least 2 points
above a signal-to-noise cut in any band. We implement flat priors on all
parameters apart from the zero points, which have strong priors.

.. math::
    P(\mu_L, \sigma_L, \mu_s, \sigma_s, Z_i, S| \mathbf{\hat{c}_i}, \mathbf{\hat{t}}, \hat{z}) \propto P(\mathbf{\hat{c}_i},\mathbf{\hat{t}},\mathbf{\hat{z}}|S, \mu_L, \sigma_L, \mu_s, \sigma_s, Z_i) P(Z_i)

Focusing on the likelihood, moving our selection criterion to the denominator,
and remember our observed data must already have passed the selection criteria, we have

.. math::
    \mathcal{L} &= \frac{P(\mathbf{\hat{c}_i}, \mathbf{\hat{t}}, \hat{z}| \mu_L, \sigma_L, \mu_s, \sigma_s, Z)}{P(S|\mu_L, \sigma_L, \mu_s, \sigma_s, Z)} \\
    &= \frac{\int dL \int dt \int ds \int dz \int dc \int df P(\mathbf{\hat{c}_i} | \mathbf{c_i}) P(\mathbf{c_i}|\mathbf{f}, Z_i) P(\mathbf{f} | \mathbf{\hat{t}},\hat{z}, L, t, s) P(L | \mu_L, \sigma_L) P(s | \mu_s, \sigma_s)}{P(S|\mu_L, \sigma_L, \mu_s, \sigma_s, Z)} \\
    &= \frac{\idotsint dL\,dt\,ds\,dz\,dc\,df\, P(\mathbf{\hat{c}_i} | \mathbf{c_i}) \delta\left(\mathbf{c_i} - \mathbf{f} 10^{Z_i/2.5}\right) \delta\left(\mathbf{f} - \hat{z}^{-2} L_0 \exp\left[-\frac{(t- \mathbf{\hat{t}})^2}{2s^2}\right]\right) P(L | \mu_L, \sigma_L) P(s | \mu_s, \sigma_s)}{P(S|\mu_L, \sigma_L, \mu_s, \sigma_s, Z)} \\
    &= \frac{\iiiint dL\,dt\,ds\,dz\, P(\mathbf{\hat{c}_i} | \mathbf{c_i}) P(L | \mu_L, \sigma_L) P(s | \mu_s, \sigma_s)}{P(S|\mu_L, \sigma_L, \mu_s, \sigma_s, Z)},

where I have removed the :math:`\delta` functions from the last line to simplify notation.




"""