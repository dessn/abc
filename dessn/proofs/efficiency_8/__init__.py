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
    &= \frac{\idotsint dL\,dt\,ds\,dc\,df\, P(\mathbf{\hat{c}_i} | \mathbf{c_i}) \delta\left(\mathbf{c_i} - \mathbf{f} 10^{Z_i/2.5}\right) \delta\left(\mathbf{f} - \hat{z}^{-2} L_0 \exp\left[-\frac{(t- \mathbf{\hat{t}})^2}{2s^2}\right]\right) P(L | \mu_L, \sigma_L) P(s | \mu_s, \sigma_s)}{P(S|\mu_L, \sigma_L, \mu_s, \sigma_s, Z)} \\
    &= \frac{\iiint dL\,dt\,ds\, P(\mathbf{\hat{c}_i} | \mathbf{c_i}) P(L | \mu_L, \sigma_L) P(s | \mu_s, \sigma_s)}{P(S|\mu_L, \sigma_L, \mu_s, \sigma_s, Z)},

where I have removed the :math:`\delta` functions from the last line to simplify notation and used
perfect measurement of redshift to remove the integral over possible redshifts.

Denoting :math:`P(S|\mu_L, \sigma_L, \mu_s, \sigma_s, Z) = W`, we can write our posterior as

.. math::
    P(\theta| \mathbf{\hat{c}_i}, \mathbf{\hat{t}}, \hat{z}) &\propto
    \frac{\iiint dL\,dt\,ds\, P(\mathbf{\hat{c}_i} | \mathbf{c_i}) P(L | \mu_L, \sigma_L) P(s | \mu_s, \sigma_s)}{W} \\
    &\propto \frac{\iiint dL\,dt\,ds\, \mathcal{N} \left(\mathbf{\hat{c}_i}; \mathbf{c_i}, \sqrt{ \mathbf{c_i}} \right)
    \mathcal{N}(L; \mu_L, \sigma_L) \mathcal{N}P(s; \mu_s, \sigma_s)}{W}

We implement the above model, running MCMC fits with :math:`W=1`. We will then compute the actual
:math:`W` to resample the biased MCMC chains and recover the correct distributions.

The model PGM is constructed as follows:

.. figure::     ../dessn/proofs/efficiency_8/output/pgm.png
    :width:     100%
    :align:     center

And is given data generated with :math:`\mu_L = 500,\ \sigma_L=50,\ \mu_s = 20,\ \sigma_s=4,\ Z_i={0}`
where each object has 20 observations, separated by a space of two days each,
in each of the two bands. A total of 400 objects are generated, and then the selection
cuts are applied, giving the following distribution:

.. figure::     ../dessn/proofs/efficiency_8/output/data.png
    :width:     80%
    :align:     center

Now we also need to determine the appropriate weights.

.. math::
    W &= P(S|\mu_L, \sigma_L, \mu_s, \sigma_s, Z)  \\
    &= \idotsint dL_0 \, dt \, ds \, dz\, d\mathbf{c_i} P(S, \mathbf{c_i}, L_0, t, s, z|\mu_L, \sigma_L, \mu_s, \sigma_s, Z) \\
    &= \idotsint dL_0 \, dt \, ds \, dz\, d\mathbf{c_i} P(S|\mathbf{c_i}) P(\mathbf{c_i}|L_0, t, \hat{t}, z, Z) P(L_0|\mu_L,\sigma_L) P(s|\mu_s,\sigma_s)
    &= \int dL_0 \int dt \int ds \int dz\int d\mathbf{c_i} P(S|\mathbf{c_i}) P(\mathbf{c_i}|L_0, t, \hat{t}, z, Z) P(L_0|\mu_L,\sigma_L) P(s|\mu_s,\sigma_s)

Note in here that we insert :math:`\hat{t}` inside the equation without an integral. This is because
the time of the observation is part of the experiment, not an observable, however we treat it
as an observable in the first section as it is given data in the same way the experimental outcome
is given data. As the weights represent the efficiency over all possible data, given the same
experiment is performed, we use the same :math:`t` values without an integral.



"""