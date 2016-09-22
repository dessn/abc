r""" My attempt at a proper STAN model.

I follow Rubin et al. (2015) with some changes:

    1. I do not model :math:`\alpha` and :math:`\beta` as functions of redshift, due to my limited dataset.
    2. I do not take outlier detection into account.
    3. I do not take zero point covariance into account.
    4. I do not take selection effects into account, *in the STAN model.*
    5. I consider :math:`w` as a free parameter, instead of just :math:`\Omega_m`.
    6. I incorporate intrinsic dispersion via SN population, not as an observational effect.

Instead, selection effects are taken into account via the methodology discussed
and verified in the model proofs section of this work. To continue, we should
first formalise the model itself.

----------

Parameters
----------

**Cosmological Parameters**:

    * :math:`\Omega_m`: matter density
    * :math:`w`: dark energy equation of state
    * :math:`\alpha`: Phillips correction for stretch
    * :math:`\beta`: Phillips correction for colour

**Population parameters**:

    * :math:`\langle M_B \rangle`: mean absolute magnitude of supernova
    * :math:`\sigma_{M_B}`: standard deviation of absolute magnitudes
    * :math:`\langle c \rangle`: mean colour
    * :math:`\sigma_c`: standard deviation of  colour
    * :math:`\langle x_1 \rangle`: mean scale
    * :math:`\sigma_{x_1}`: standard deviation of scale
    * :math:`\rho`: correlation (matrix) between absolute magnitude, colour and stretch

.. warning::
    Not currently implemented:

    **Marginalised Parameters**:
        * :math:`\delta(0)` and :math:`\delta(\infty)`: The magnitude-mass relationship

----------

Model
-----

.. note::
    In this section I will briefly outline the model I am using. For the
    methodology used to incorporate selection effects, I recommend reading through the
    simpler and more detailed examples provided at:

        * :ref:`efficiency1`
        * :ref:`efficiency2`
        * :ref:`efficiency3`

    There are more examples that can be found at :ref:`proofs`, however the first few
    should be sufficient.

We wish to model our posterior, given our observations, our model :math:`\theta`, and
selection effects :math:`S`. Our observations are the light curves themselves,
the summary statistics that result from them :math:`\lbrace \hat{m_B}, \hat{c}, \hat{x_1} \rbrace`,
the covariance for the summary statistics :math:`\hat{C}`, the redshifts of the
objects :math:`\hat{z}` and a normalised mass estimate :math:`\hat{m}`. We thus signify
observed variables with the hat operator. In this work we will be modelling
:math:`\lbrace \hat{m_B}, \hat{c}, \hat{x_1} \rbrace` as having true underlying
values, however assume that  :math:`\hat{z}` and :math:`\hat{m}` are
known (:math:`\hat{z} = z,\ \hat{m}=m)`.

.. math::
    P(\theta S|d) &\propto P(d|S\theta) P(\theta)

Let us separate out the selection effects:

.. math::
    P(\theta S|d) &\propto  \frac{P(d,S|\theta) P(\theta)}{P(S|\theta)}   \\[10pt]
    &\propto \frac{P(S|d,\theta) P(d|\theta) P(\theta)}{P(S|\theta)}

As our data must have passed the selection cuts, by definition, the numerator
reduces down.

.. math::
    P(\theta S|d) &\propto  \frac{P(d|\theta)P(\theta)}{P(S|\theta)}

STAN Model
~~~~~~~~~~

Let us examine only the numerator for the time being. The numerator is the model
which ends up implemented in STAN, whilst the denominator must be implemented
differently. For simplicity, let us denote the population parameters
:math:`\langle M_B \rangle... \rho` shown under the Population header as :math:`\gamma`.

.. math::
    P(d|\theta)P(\theta) &= P(\hat{m_B}, \hat{x_1}, \hat{c}, \hat{z}, \hat{m} |
    \Omega_m, w, \alpha, \beta, \gamma)
    P(\Omega_m, w, \alpha, \beta, \gamma)

Now, let us quickly deal with the priors so I don't have to type them out again and again.
We will treat :math:`\sigma_{M_B},\ \sigma_{x_1},\, \sigma_c`
with Cauchy priors, :math:`\rho` with an LKJ prior, and other parameters with flat priors.
So now we can focus on the likelihood's numerator, which is

.. math::
    \mathcal{L} &= P(\hat{m_B}, \hat{x_1}, \hat{c}, \hat{z}, \hat{m} |
    \Omega_m, w, \alpha, \beta, \gamma) \\[10pt]
    &= \int dm_B \int dx_1 \int dc \  P(\hat{m_B}, \hat{x_1}, \hat{c}, \hat{z}, \hat{m}, m_B, x_1, c |
    \Omega_m, w, \alpha, \beta, \gamma) \\[10pt]
    &= \int dm_B \int dx_1 \int dc \  P(\hat{m_B}, \hat{x_1}, \hat{c} | m_B, x_1, c) P(m_b, x_1, c, \hat{z}, \hat{m}|
    \Omega_m, w, \alpha, \beta, \gamma) \\[10pt]
    &= \int dm_B \int dx_1 \int dc \  \mathcal{N}\left( \lbrace \hat{m_B}, \hat{x_1}, \hat{c} \rbrace | \lbrace m_B, x_1, c \rbrace, C \right)
    P(m_b, x_1, c, \hat{z}, \hat{m}| \Omega_m, w, \alpha, \beta, \gamma)

Now, in order to calculate :math:`P(m_b, x_1, c, \hat{z}, \hat{m}| \Omega_m, w, \alpha, \beta, \gamma)`,
we need to transform from :math:`m_B to M_B`. This is done via the following maths:

.. math::
    M_B = m_B + \mu + \alpha x_1 - \beta c

.. + k(z) m

where we define :math:`\mu` as

.. and :math:`k(z)` as

.. math::
    \mu &= 5 \log_{10} \left[ \frac{(1 + z)c}{H_0 \times 10{\rm pc}} \int_0^z \left(
    \Omega_m (1 + z)^3 + (1 - \Omega_m)(1+z)^{3(1+w)} \right) \right] \\[10pt]

.. k(z) &= \delta(0) \left[\frac{1.9\left( 1 - \frac{\delta(\infty)}{\delta(0)}
    \right)}{0.9 + 10^{0.95z}} + \frac{\delta(\infty)}{\delta(0)} \right]

Thus :math:`M_B` is a function of :math:`\Omega_m, w, \alpha, \beta, x_1, c, z`.

----------

**Distributions**:

Latent colour is drawn from a skew normal. Latent stretch is drawn from a normal.

.. math::
    P(c| \langle c \rangle, \sigma_c, \alpha_c) &= \mathcal{N}_{\rm skew}
    (c|\langle c \rangle, \sigma_c, \alpha_c) \\

    P(x_1| \langle x_1 \rangle, \sigma_{x1}) &= \mathcal{N}(x_1|\langle  x_1 \rangle, \sigma_{x1})

**Mass correction**:

We continue to follow Rubin et al. (2015) and model the mass correction as

.. math::
    \Delta M = \hat{m} k(z) = \hat{m} \, \delta(0) \left[\frac{1.9\left( 1 - \frac{\delta(\infty)}{\delta(0)}
    \right)}{0.9 + 10^{0.95z}} + \frac{\delta(\infty)}{\delta(0)} \right]

**Dispersion**:

Dispersion is modelled by adjusting the covariance matrix between the
model :math:`\lbrace m_B,x_1,c \rbrace` and the observed. The extra factor
added to the covariance is

.. math::
    \sigma^2_{\rm add} = \sigma^2_{\rm int} \times \begin{pmatrix} f_m &
    \rho_{12}\frac{\sqrt{f_m f_{x1}}}{0.13} & \rho_{13} \frac{\sqrt{f_m f_c}}{-3.0} \\
    \rho_{12} \frac{\sqrt{f_m f_{x1}}}{0.13} & \frac{f_{x1}}{0.13^2} &
    \rho_{23} \frac{\sqrt{f_{x1} f_{c}}}{(-3.0)(0.13)} \\
    \rho_{13} \frac{\sqrt{f_m f_c}}{-3.0} & \rho_{23}
    \frac{\sqrt{f_{x1} f_c }}{(-3.0)(0.13)} & \frac{f_c}{(-3.0)^2}    \end{pmatrix}

**Combined**:

Whilst it is tempting to write out everything with all variables clearly displayed, I
am going to pass. But once, for posterity, we want to calculate:

.. math::
    P(d|\theta) = P(\hat{m_B},\hat{x_1},\hat{c}, \hat{z}, \hat{m}|\Omega_M, w, \alpha, \beta,
    \langle c \rangle, \sigma_c, \alpha_c, \langle x_1 \rangle, \sigma_{x_1}, M_B,
    \sigma_{\rm int}, f_{m},\ f_{x_1},\ f_{c}, \rho, \delta(0), \delta(\infty))

Let us introduce several transformations. Firstly, from cosmology to distance modulus, such that

.. math::
    \mu = 5 \log_{10} \left[ \frac{(1 + z)c}{H_0 \times 10{\rm pc}} \int_0^z \left(
    \Omega_m (1 + z)^3 + (1 - \Omega_m)(1+z)^{3(1+w)} \right) \right]

Secondly, the predicted apparent magnitudes :math:`m_B` can be constructed
using the distance modulus, the Phillips relation, and the mass correction.

.. math::
    m_B = M - \mu - \alpha x_1 + \beta c - \Delta M

Putting these together, we have that

.. math::
    P(d|\theta) = \prod_{\rm obs} \mathcal{N} \left(\lbrace \hat{m_B}, \hat{c}, \hat{x_1} \rbrace |
     \lbrace {m_B}, {c}, {x_1},  \rbrace, C + \sigma_{\rm add}^2\right)
     \mathcal{N}_{\rm skew}(c|\langle c \rangle, \sigma_c, \alpha_c)
     \mathcal{N}(x_1|\langle  x_1 \rangle, \sigma_{x1})

Looking now at the priors, we implement :math:`{\rm Cauchy}(0,2.5)` priors on
:math:`\sigma_{\rm int}`, :math:`\sigma_{x_1}` and :math:`\sigma_c` and flat priors on
all other variables.

.. math::
    P(d|\theta)P(\theta) &\propto
    \rm{Cauchy}(\sigma_{\rm int}|0,2.5)
    \rm{Cauchy}(\sigma_{x_1}|0,2.5)
    \rm{Cauchy}(\sigma_{c}|0,2.5) \\
    &\quad\quad\quad \prod_{\rm obs} \mathcal{N} \left(\lbrace \hat{m_B}, \hat{c}, \hat{x_1} \rbrace |
     \lbrace {m_B}, {c}, {x_1},  \rbrace, C + \sigma_{\rm add}^2\right)
     \mathcal{N}_{\rm skew}(c|\langle c \rangle, \sigma_c, \alpha_c)
     \mathcal{N}(x_1|\langle  x_1 \rangle, \sigma_{x1})

A very rough fit for this (defintiely not enough steps), is shown below, for a thousand generated
supernova.

.. figure::     ../dessn/models/d_simple_stan/output/plot.png
    :align:     center


Selection Effects
~~~~~~~~~~~~~~~~~

Having formulated a probabilistic model for the numerator of our posterior (and sent it off
to STAN), we can now turn our attention to the denominator: :math:`P(S|\theta)`. In English,
this is the probability of a possible observed data sets passing the selection effects, integrated
over all possible observations. It is also equivalent to the normalisation condition
of our likelihood! Now, for :math:`P(S|\theta)` to make mathematical sense, the selection
effects :math:`S` need to apply onto some data:

.. math::
    P(S|\theta) = \int dR P(R,S|\theta)

where :math:`R` is a potential realisation of our experiment. To write this down,
and taking into account we can model supernova such that we can determine
the efficiency as a function of constructed :math:`\lbrace m_B, x_1, c \rbrace`, we
have:

.. math::
    P(S|\theta) &= \int d\hat{m_B} \int d\hat{x}_1 \int d\hat{c}
    \int dz \int dm \int dm_B \int dx_1 \int dc \
    P(\hat{m_B}, m_B, \hat{x}_1, x_1, \hat{c}, c, z, m, S|\theta) \\
    &= \int d\hat{m_B} \int d\hat{x}_1 \int d\hat{c}
    \int dz \int dm \int dm_B \int dx_1 \int dc \
    P(\hat{m_B}, \hat{x}_1, \hat{c} | m_B, x_1, c) P(m_B, x_1, c, z, m, S|\theta) \\
    &= \idotsint d\hat{m_B}\, d\hat{x}_1 \, d\hat{c} \, dz \, dm \, dm_B \, dx_1 \, dc \
    P(\hat{m_B}, \hat{x}_1, \hat{c} | m_B, x_1, c) P(S|m_B, x_1, c) P(m_B, x_1, c, z, m|\theta) \\
    &= \idotsint d\hat{m_B}\, d\hat{x}_1 \, d\hat{c} \, dz \, dm \, dm_B \, dx_1 \, dc \
    P(\hat{m_B}, \hat{x}_1, \hat{c} | m_B, x_1, c) P(S|m_B, x_1, c, z) P(c|\theta) P(x_1 | \theta) P(m_B, z, m|\theta) \\

Note again that we assume redshift and mass are perfectly known, so relationship between
actual (latent) redshift and mass and the observed quantity is a delta function, hence why
they only appear once in the equation above.

As we integrate over all possible realisations, we have that over all space
:math:`P(\hat{m_B}, \hat{x}_1, \hat{c} | m_B, x_1, c) = \iiint_{-\infty}^{\infty} \mathcal{N}( \hat{m_B}, \hat{x}_1, \hat{c} | m_B, x_1, c, \sqrt{\sigma^2_{\rm obs} + \sigma^2_{\rm add}})= 1`,
and as such we can remove it from the integral. We also note that at the moment the model
not contain any details of the mass distribution of galaxies, which may be an issue.

.. math::
    P(S|\theta) &= \idotsint dz \, dm \, dm_B \, dx_1 \, dc  P(S|m_B, x_1, c, z) P(m_B|z, m, x_1, c, \theta) P(z) P(m) P(c|\theta) P(x_1 | \theta) \\

Addressing each component individually:

.. math::
    P(z)&= \text{Redshift distribution from DES volume}\\
    P(m) &= \text{Unknown mass distribution} \\
    P(c|\theta) &= P(c| \langle c \rangle, \sigma_c, \alpha_c) = \mathcal{N}_{\rm skew}(c|\langle c \rangle, \sigma_c, \alpha_c) \\
    P(x_1|\theta) &= P(x_1| \langle x_1 \rangle, \sigma_{x1}) = \mathcal{N}(x_1|\langle  x_1 \rangle, \sigma_{x1}) \\
    P(m_B|z, m, x_1, c, \theta) &= \delta(m_B, \Omega_m, w, z, x_1, c, m, \alpha, \beta, M_B) \\
    P(S|m_B, x_1, c, z) &= \text{Ratio of SN generated that pass selection cuts for given SN parameters}

Now enter the observational specifics of our survey: how many bands, the band passes,
frequency of observation, weather effects, etc. The selection effects we need to model are

    * At least 5 epochs between :math:`-99 < t < 60`.
    * :math:`0.0 < z < 1.2`.
    * At least one point :math:`t < -2`.
    * At least one point :math:`t > 10`.
    * At least 2 filters with :math:`S/N > 5`.

These cuts there require us to model the light curves themselves, not just
the magnitude, colour and stretch distribution. We thus need to be able to
go from a given :math:`\lbrace m_B, x_1, c, z, m\rbrace`, to a supernova
light curve.

-------

    **Technical aside**: Calculating :math:`\delta(m_B, \Omega_m, w, z, x_1, c, m \alpha, \beta, M_B)`
    is not an analytic task. It has complications not just in the distance modulus being the
    result of an integral, but also that the colour and stretch correction factors make
    extra use of supernova specific values. The way to efficiently determine :math:`P(m_B|...)`
    is given as follows:

        1. Draw samples of :math:`z` from the DES redshift distribution.
        2. Using :math:`\Omega_m` and :math:`w` to translate this to a :math:`\mu` distribution.
        3. Take simulated supernova :math:`m_B, x_1, c, m, z` values, and use :math:`\alpha, \beta, \delta(0), \delta(\infty), M_B` to get :math:`\mu^* = m_B + \alpha x_1 - \beta c + m k(z) - M_B`
        4. Using :math:`P(\mu)` from step 2, determine :math:`P(-\mu^*)`, which is equivalent to :math:`P(m_B|z, m, x_1, c, \theta)`.

    However, this breaks down because we move dispersion into the observation and thus remove it from the integral.
    This means that :math:`P(\mu)` remains a delta function, and thus :math:`P(m_B|z, m, x_1, c, \theta) = 0` for
    all supernova. So what we realistically need is a better way of modelling the magnitude, colour and stretch
    population. One way is to do it as a multivariate normal, but then we lose the ability to skew the colour
    distribution. I cannot find a way of doing a multivariate skew distribution. Perhaps we could manually
    combine them. Regardless, I cannot see how modelling populations *and* intrinsic dispersion is the correct
    method to go with, because your population should really encompass the dispersion in the actual events.

    I need to address this point before continuing. I feel the easiest thing to do at the moment is to
    move to a multivariate normal model of :math:`\lbrace m_B, x_1, c \rbrace`.

-------



"""