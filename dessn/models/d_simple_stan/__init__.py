r""" My attempt at a proper STAN model.

I follow Rubin et al. (2015) with some changes:

    1. I do not model :math:`\alpha` and :math:`\beta` as functions of redshift, due to my limited dataset.
    2. I do not take outlier detection into account.
    3. I do not take zero point covariance into account.
    4. I do not take selection effects into account, *in the STAN model.*
    5. I consider :math:`w` as a free parameter, instead of just :math:`\Omega_m`.

Instead, selection effects are taken into account via the methodology discussed
and verified in the model proofs section of this work. To continue, we should
first formalise the model itself.

Parameters
----------

**Cosmological Parameters**:

    * :math:`\Omega_m`: matter density
    * :math:`w`: dark energy equation of state
    * :math:`\alpha`: Phillips correction for stretch
    * :math:`\beta`: Phillips correction for colour

**Hyperparameters**:

    * :math:`\langle c \rangle`: colour distribution mean
    * :math:`\sigma_c`: colour distribution standard deviation
    * :math:`\alpha_c`: colour distribution skewness
    * :math:`\langle x_1 \rangle`: scale distribution mean
    * :math:`\sigma_{x_1}`: scale distribution standard deviation

**Marginalised Parameters**:

    * :math:`M_B`: absolute magnitude of type Ia supernova
    * :math:`\sigma_{\rm int}`: unexplained/intrinsic dispersion of supernova
    * :math:`f_{m},\ f_{x_1},\ f_{c}`: simplex fraction of intrinsic dispersion accorded to magnitude, stretch and colour
    * :math:`\rho`: correlation of intrinsic dispersion model, used in conjuction with parameters above
    * :math:`\delta(0)` and :math:`\delta(\infty)`: The magnitude-mass relationship

Model
-----

You may need a wide screen for this section.

Specifically, we wish to model our posterior, given our
observations :math:`x`, our model :math:`\theta`, and selection effects :math:`S`. Our
observations are the light curves themselves, the summary statistics that result from them
:math:`\lbrace \hat{m_B}, \hat{c}, \hat{x_1} \rbrace`, the covariance for the summary statistics
:math:`C`, the redshifts of the objects :math:`\hat{z}` and a
normalised mass estimate :math:`\hat{m}`.

Note that in this work we will be modelling :math:`\lbrace \hat{m_B}, \hat{c}, \hat{x_1} \rbrace`
as having true underlying values, however assume that  :math:`\hat{z}`
and :math:`\hat{m}` are known (:math:`\hat{z} = z,\ \hat{m}=m)`.

.. math::
    P(\theta|d) \propto P(d|S\theta) P(\theta)

Let us separate out the selection effects and focus on the likelihood.

.. math::
    \mathcal{L} = \frac{P(d,S|\theta)}{P(S|\theta)}

    \mathcal{L} = \frac{P(S|d,\theta) P(d|\theta)}{P(S|\theta)}

As our data must have passed the selection cuts, by definition, the numerator
reduces down.

.. math::
    \mathcal{L} = \frac{P(d|\theta)}{P(S|\theta)}

STAN Model
~~~~~~~~~~

Let us examine only the numerator for the time being. The numerator is the model
which ends up implemented in STAN, whilst the denominator must be implemented
differently. To try and make the maths more digestible, let us split up components:

**Distributions**:

Latent colour is drawn from a skew normal. Latent stretch is drawn from a normal.

.. math::
    P(c| \langle c \rangle, \sigma_c, \alpha_c) &= \mathcal{N}_{\rm skew}
    (c|\langle c \rangle, \sigma_c, \alpha_c) \\

    P(x_1| \langle x_1 \rangle, \sigma_{x1}) &= \mathcal{N}(x_1|\langle  x_1 \rangle, \sigma_{x1})

**Mass correction**:

We continue to follow Rubin et al. (2015) and model the mass correction as

.. math::
    \Delta M = \hat{m} \delta(0) \left[\frac{1.9\left( 1 - \frac{\delta(\infty)}{\delta(0)}
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
    P(\hat{m_B}, \hat{x}_1, \hat{c} | m_B, x_1, c) P(S|m_B, x_1, c) P(c|\theta) P(x_1 | \theta) P(m_B, z, m|\theta) \\

Note again that we assume redshift ans mass are perfectly known, so relationship between
actual (latent) redshift and mass and the observed quantity is a delta function, hence why
they only appear once in the equation above.

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

--------

Now at this point, I am stuck. In general, I want to know the efficiency
(ratio of observed sn to ratio of all sn) for each cosmology. There are several
ways that one could potentially do this:

**Full SNANA**

    Run a full SNANA simulation for each input cosmology. Assumes things like a dispersion
    model as given above exist in SNANA. Also would require 1 sim per step, which may be slow.
    Also involves splitting the project (no press enter and it runs start to finish).
    Have a partial python interface thanks to Elise and Rachel, but would have to adapt.

**Pre-gen**

    If I can model the chance of detecting a SN as a function of m_B, x_1 and c (and z), I
    can pre-generate a host of potential supernova, and then as a function of m_B, x_1 and c
    determine the probability of them passing selection cuts. I could then use the cosmology and
    hyperparameters to draw from this pool of supernova to determine efficiency. This would
    be faster (one big sim and then sample from it), but would involve running a sim without
    cuts (danger), so that I can actually calculate the chance of a particular m_B, x_1, c
    passing the cuts. Would also then need a way of coming up with the m_B distribution. If
    I can get a redshift distribution, I could combine that with cosmology, dispersion (etc)
    and come up with m_B samples from the right distribution, but I worry this would be
    different to the m_B distribution I would get if I just used SNANA.

Now, have been trying to the second option, however there are too many effects
not taken into account by sncosmo when I create mock supernova, such that I get almost
perfect efficiency for the deep fields up to z=1.2. Things like weather, epochs are done
too roughly, and extinction is just not taken into account at all.



"""