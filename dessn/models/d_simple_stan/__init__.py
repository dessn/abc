r""" My attempt at a proper STAN model.

I follow Rubin et al. (2015) with some changes:

    1. I do not model :math:`\alpha` and :math:`\beta` as functions of redshift, due to my limited dataset.
    2. I do not take outlier detection into account.
    3. I do not take zero point covariance into account.
    4. I incorporate intrinsic dispersion via SN population, not as an observational effect.

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
selection effects :math:`S_D`, where the selection effect acts on a data sample :math:`D`.
Our specific observations :math:`d` are the light curves themselves,
the summary statistics that result from them :math:`\lbrace \hat{m_B}, \hat{c}, \hat{x_1} \rbrace`,
the covariance for the summary statistics :math:`\hat{C}`, the redshifts of the
objects :math:`\hat{z}` and a normalised mass estimate :math:`\hat{m}`. We thus signify
observed variables with the hat operator. In this work we will be modelling
:math:`\lbrace \hat{m_B}, \hat{c}, \hat{x_1} \rbrace` as having true underlying
values, however assume that  :math:`\hat{z}` and :math:`\hat{m}` are
known :math:`(\hat{z} = z,\ \hat{m}=m)`.

Note that we denote a data sample here as :math:`D`, which is distinct to the experimental
data :math:`d`. :math:`D` is simply our model stating that *any* input data must
satisfy the selection criterion.

.. math::
    :label: a

    P(\theta, S_D|D=d) &= \frac{ P(D=d|S_D,\theta) P(\theta) }{\int P(D=d|S_D,\theta^\prime) P(\theta^\prime)  \,d\theta^\prime}  \\[10pt]


Let us separate out the selection effects:

.. math::
    :label: b

    P(\theta, S_D|D=d) &= \frac{P(D=d,S_D|\theta) P(\theta)}{P(S_D|\theta) \int  P(D=d|S_D,\theta^\prime) P(\theta^\prime) \,d\theta^\prime}  \\[10pt]
    &= \frac{P(S_D|D=d,\theta) P(D=d|\theta) P(\theta)}{P(S_D|\theta) \int  P(D=d|S_D,\theta^\prime) P(\theta^\prime) \,d\theta^\prime}

As normal, we can see that our posterior is normalised over all possible :math:`\theta`. Now, in order for
our selection function in the denominator to act on a given data sample - which it must to be evaluable, we
introduce an integral over all possible data realisations.

.. math::
    :label: bg

    P(\theta, S_D|D=d) &= \frac{P(S_D|D=d,\theta) P(D=d|\theta) P(\theta)}{\int  P(S_D, D=R|\theta) \,dR \int  P(D=d|S_D,\theta^\prime) P(\theta^\prime)\,d\theta^\prime} \\[10pt]
    P(\theta, S_D|D=d) &= \frac{P(S_D|D=d,\theta) P(D=d|\theta) P(\theta)}{\int  P(S_D | D=R, \theta) P(D=R|\theta) \,dR\int  P(D=d|S_D,\theta^\prime) P(\theta^\prime)\,d\theta^\prime} \\


As our data must have passed the selection cuts, by definition, the numerator reduces down via :math:`P(S_D|D=d,\theta) = 1`.

.. math::
    :label: c

    P(\theta, S_D|D=d) &= \frac{P(D=d|\theta) P(\theta)}{\int  P(S_D | D=R, \theta) P(D=R|\theta) \,dR \int  P(D=d|S_D,\theta^\prime) P(\theta^\prime)\,d\theta^\prime} \\

Finally, as :math:`\int  P(D=d|S_D,\theta\prime) P(\theta\prime)\,d\theta\prime` does not depend on :math:`\theta`, we
remove it and replace the equality with a proportionality constraint

.. math::
    :label: cg

    P(\theta, S_D|D=d) &= \frac{P(D=d|\theta) P(\theta)}{\int  P(S_D | D=R, \theta) P(D=R|\theta) \,dR } \\


To simplify notation, I will refer to the :math:`P(D=d|\theta)` as the likelihood :math:`\mathcal{L}`, and the denominator as the bias :math:`B`

.. math::
    :label: ch

    P(\theta, S_D|D=d) &= \frac{\mathcal{L} P(\theta)}{B} \\


----------

STAN Model
~~~~~~~~~~

Let us examine only the numerator for the time being. The numerator is the model
which ends up implemented in STAN, whilst the denominator must be implemented
differently. For simplicity, let us denote the population parameters
:math:`\lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle, \sigma_{M_B}, \sigma_{x_1}, \sigma_c, \rho \rbrace`
shown under the Population header as :math:`\gamma`.

.. math::
    :label: d

    \mathcal{L} P(\theta) &= P(\hat{m_B}, \hat{x_1}, \hat{c}, \hat{z}, \hat{m} |
    \Omega_m, w, \alpha, \beta, \gamma)
    P(\Omega_m, w, \alpha, \beta, \gamma) \\

Now, let us quickly deal with the priors so I don't have to type them out again and again.
We will treat :math:`\sigma_{M_B},\ \sigma_{x_1},\, \sigma_c`
with Cauchy priors, :math:`\rho` with an LKJ prior, and other parameters with flat priors.
So now we can focus on the likelihood's numerator, which is

.. math::
    :label: e

    \mathcal{L} &= P(\hat{m_B}, \hat{x_1}, \hat{c}, \hat{z}, \hat{m} |
    \Omega_m, w, \alpha, \beta, \gamma) \\[10pt]
    &= \int dm_B \int dx_1 \int dc \  P(\hat{m_B}, \hat{x_1}, \hat{c}, \hat{z}, \hat{m}, m_B, x_1, c | \Omega_m, w, \alpha, \beta, \gamma) \\[10pt]
    &= \int dm_B \int dx_1 \int dc \  P(\hat{m_B}, \hat{x_1}, \hat{c}, z, m, m_B, x_1, c | \Omega_m, w, \alpha, \beta, \gamma) \\[10pt]

Where in the last line I have used the fact that we assume mass and redshift are precisely known.
Also, as we assume that the observed summary statistics :math:`\hat{m_B}, \hat{x_1}, \hat{c}` are normally
distributed around the true values :math:`m_B,x_1,c`, we can separate them out:

.. math::
    :label: eg

    \mathcal{L} &= \int dm_B \int dx_1 \int dc \  P(\hat{m_B}, \hat{x_1}, \hat{c} | m_B, x_1, c, z, m, \Omega_m, w, \alpha, \beta, \gamma) P(m_B, x_1, c, z, m| \Omega_m, w, \alpha, \beta, \gamma) \\[10pt]
    &= \int dm_B \int dx_1 \int dc \  P(\hat{m_B}, \hat{x_1}, \hat{c} | m_B, x_1, c) P(m_B, x_1, c, z, m| \Omega_m, w, \alpha, \beta, \gamma) \\[10pt]
    &= \int dm_B \int dx_1 \int dc \  \mathcal{N}\left( \lbrace \hat{m_B}, \hat{x_1}, \hat{c} \rbrace | \lbrace m_B, x_1, c \rbrace, C \right) P(m_B, x_1, c, z, m| \Omega_m, w, \alpha, \beta, \gamma)

Now, in order to calculate :math:`P(m_B, x_1, c, \hat{z}, \hat{m}| \Omega_m, w, \alpha, \beta, \gamma)`,
we need to transform from :math:`m_B` to :math:`M_B`. We transform using the following relationship:

.. math::
    :label: f

    M_B = m_B - \mu + \alpha x_1 - \beta c + k(z) m

where we define :math:`\mu` as


.. math::
    :label: g

    \mu &= 5 \log_{10} \left[ \frac{(1 + z)c}{H_0 \times 10{\rm pc}} \int_0^z \left(
    \Omega_m (1 + z)^3 + (1 - \Omega_m)(1+z)^{3(1+w)} \right) \right] \\

and :math:`k(z)` as

.. math::
    :label: h

    k(z) &= \delta(0) \left[ \frac{1.9\left( 1 - \frac{\delta(\infty)}{\delta(0)}
    \right)}{0.9 + 10^{0.95z}} + \frac{\delta(\infty)}{\delta(0)} \right] \\

Thus :math:`M_B` is a function of :math:`\Omega_m, w, \alpha, \beta, x_1, c, z`. Or, more probabilistically,

.. math::
    P(M_B, m_B) = \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right).

We can thus introduce a latent variable :math:`M_B` and immediately remove the :math:`m_B` integral via the delta function.

.. math::
    :label: i

    \mathcal{L} &= \int dm_B \int dx_1 \int dc \int M_B \  \mathcal{N}\left( \lbrace \hat{m_B}, \hat{x_1}, \hat{c} \rbrace | \lbrace m_B, x_1, c \rbrace, C \right) P(m_B, M_B, x_1, c, z, m| \Omega_m, w, \alpha, \beta, \gamma) \\[10pt]

.. math::
    :label: ig

    P(m_B, M_B, x_1, c, z, m| \theta) &= P(m_B | M_B, x_1, c, z, m \Omega_m, w, \alpha, \beta, \gamma) P (M_B, x_1, c, z, m \Omega_m, w, \alpha, \beta, \gamma | \Omega_m, w, \alpha, \beta, \gamma)  \\[10pt]
    &= \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) P (M_B, x_1, c, z, m \Omega_m, w, \alpha, \beta, \gamma | \Omega_m, w, \alpha, \beta, \gamma)  \\[10pt]
    &= \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) P (M_B, x_1, c, | \gamma) P(z) P(m)  \\[10pt]
    &= \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) \mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right) P(z) P(m) \\[10pt]

where

.. math::
    :label: j

    V &= \begin{pmatrix}
    \sigma_{M_B}^2                        & \rho_{12} \sigma_{M_B} \sigma_{x_1}         & \rho_{13} \sigma_{M_B} \sigma_{c}  \\
    \rho_{21} \sigma_{M_B} \sigma_{x_1}           & \sigma_{x_1}^2                    & \rho_{23} \sigma_{x_1} \sigma_{c}  \\
    \rho_{31} \sigma_{M_B} \sigma_{c}          & \rho_{32} \sigma_{x_1} \sigma_{c}       & \sigma_{c}^2  \\
    \end{pmatrix}

and :math:`P(z)` is the DES specific redshift distribution, and :math:`P(m)` is the mass distribution (currently assumed to not be a function cosmology and
just set to a uniform distribution, but this will need to be updated).


.. note::
    In this implementation there is no skewness in the colour distribution.
    As we do not require normalised probabilities, we can simply add in correcting
    factors (such as an additional linear probability for colour) that can emulate skewness.

Putting this back together, we now have a simple hierarchical multi-normal model.
Adding in the priors, and taking into account that we observe multiple supernova, we have
that a final numerator of:

.. math::
    :label: k

    P(\theta, S_D|D=d) &\propto
    \int dm_B \int dx_1 \int dc \int M_B\
    \rm{Cauchy}(\sigma_{M_B}|0,2.5)
    \rm{Cauchy}(\sigma_{x_1}|0,2.5)
    \rm{Cauchy}(\sigma_{c}|0,2.5)
    \rm{LKJ}(\rho|4) \\
    &\quad\quad\quad \mathcal{N}\left( \lbrace \hat{m_B}, \hat{x_1}, \hat{c} \rbrace | \lbrace m_B, x_1, c \rbrace, C \right)
    \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) \\
    &\quad\quad\quad \mathcal{N}\left( \lbrace M_B, x_1, c \rbrace |
    \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right) P(z) P(m)

A rough fit for this, is shown below, for two hundred generated supernova.

.. figure::     ../dessn/models/d_simple_stan/output/plot_full.png
    :align:     center

--------

Selection Effects
~~~~~~~~~~~~~~~~~

Having formulated a probabilistic model for the numerator of our posterior (and sent it off
to STAN), we can now turn our attention to the denominator: :math:`P(S|\theta)`. In English,
this is the probability of a possible observed data sets passing the selection effects, integrated
over all possible observations.

.. math::
    :label: l

    B = \int  P(S_D | D=R, \theta) P(D=R|\theta) \,dR

where :math:`R` is a potential realisation of our experiment. As the bias correction is not data dependent,
but model parameter dependent (cosmology dependent), the correction for each data point is identical, such
that

.. math::
    :label: lg

    B = w^N,

where we observe :math:`N` supernova. *We note here that this is actually not 100% correct, but as we follow
a Monte Carlo approach to simulate supernova using actual DES observing conditions,
the different weather and observing conditions for each supernova are taken into account statistically.*

Looking at the correction for a single supernova, and thus a single set of summary statistics and a single redshift (etc),
we follow a very similar mathematical approach as seen above in the likelihood calculation. We also
assume that selection effects can be determined as a function of apparent magnitude, colour, stretch, redshift and mass.

.. math::
    :label: m

    w &= \int d\hat{m_B} \int d\hat{x}_1 \int d\hat{c}
    \int dz \int dm \int dm_B \int dx_1 \int dc \int dM_B\
    P(\hat{m_B}, m_B, \hat{x}_1, x_1, \hat{c}, c, z, m, M_B, S_D|\theta) \\[10pt]
    &= \int d\hat{m_B} \int d\hat{x}_1 \int d\hat{c}
    \int dz \int dm \int dm_B \int dx_1 \int dc \int dM_B\
    P(\hat{m_B}, \hat{x}_1, \hat{c} | m_B, x_1, c) P(m_B, x_1, c, z, m, M_B, S_D|\theta) \\[10pt]
    &= \idotsint d\hat{m_B}\, d\hat{x}_1 \, d\hat{c} \, dz \, dm \, dm_B \, dx_1 \, dc \, dM_B\
    P(\hat{m_B}, \hat{x}_1, \hat{c} | m_B, x_1, c) P(S_D|m_B, x_1, c, z, m) P(m_B, x_1, c, z, m, M_B|\theta) \\[10pt]
    &= \idotsint d\hat{m_B}\, d\hat{x}_1 \, d\hat{c} \, dz \, dm \, dm_B \, dx_1 \, dc \, dM_B\
    P(S|m_B, x_1, c, z, m) P(\hat{m_B}, \hat{x}_1, \hat{c} | m_B, x_1, c) \\
    &\quad\quad\quad \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right)
    P(M_B, x_1, c | \gamma) P(z) P(m)


Again that we assume redshift and mass are perfectly known, so the relationship between
actual (latent) redshift and mass and the observed quantity is a delta function, hence why
they only appear once in the equation above. The important assumption
in the last line fo the equation is that the detection efficiency is to good approximation
captured by the apparent magnitude, colour, stretch, mass and redshift of the supernova. Note that both the
conditional probabilities left in the above equation were modelled as normal distributions in the likelihood.

As we integrate over all possible realisations, we have that over all space we have

.. math::
    :label: n

    P(\hat{m_B}, \hat{x}_1, \hat{c} | m_B, x_1, c) =
    \iiint_{-\infty}^{\infty} d\hat{m_B} d\hat{x_1} d\hat{c}\
    \mathcal{N}(\lbrace \hat{m_B}, \hat{x}_1, \hat{c} \rbrace | \lbrace m_B, x_1, c \rbrace, C) = 1

and as such we can remove it from the integral. As is expected, the final weight looks exactly like our likelihood,
except with some extra integral signs that marginalise over all possible experimental realisations:

.. math::
    :label: o

    w &= \idotsint dz \, dm \, dm_B \, dx_1 \, dc \, dM_B\
    P(S|m_B, x_1, c, z, m) \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) P(M_B, x_1, c | \gamma) P(z) P(m) \\

Addressing each component individually:

.. math::
    :label: p

    P(z)&= \text{Redshift distribution from DES volume}\\
    P(m) &= \text{Unknown mass distribution} \\
    P(M_B, x_1, c|\gamma) &= \mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right) \\
    P(S|m_B, x_1, c, z, m) &= \text{Ratio of SN generated that pass selection cuts for given SN parameters} \\
    \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) &= \text{Transformation function}

Now enter the observational specifics of our survey: how many bands, the band passes,
frequency of observation, weather effects, etc. The selection effects we need to model are

    * At least 5 epochs between :math:`-99 < t < 60`.
    * :math:`0.0 < z < 1.2`.
    * At least one point :math:`t < -2`.
    * At least one point :math:`t > 10`.
    * At least 2 filters with :math:`S/N > 5`.


.. note::
    :class: green

    **Technical aside**: Calculating this correction
    is not an analytic task. It has complications not just in the distance modulus being the
    result of an integral, but also that the colour and stretch correction factors make
    extra use of supernova specific values. The way to efficiently determine the efficiency
    is given as follows:

        1. Initially run a large DES-like simulation, recording all generated SN parameters and whether they pass the cuts.
        2. Using input cosmology to translate :math:`m_B, x_1, c` distribution to a :math:`M_B, x_1, c` distribution.
        3. Perform Monte-Carlo integration using the distribution. The value is :math:`P(S|m_B,x_1,c,z,m) = 1.0` if detected, :math:`0` otherwise, weighted by the probability of :math:`M_B,x_1,c,z,m` for that cosmology.

    To go into the math, our Monte Carlo integration for the weights. Our initial sample
    of supernova simulated is drawn from the multivariate normal distribution :math:`\mathcal{N}_{\rm sim}`.

    .. math::
        P(S|\theta) &= w^N \\
        &= \left[ \frac{1}{N_{\rm sim}} \sum  P(S|m_B, x_1, c, z,m)  \frac{\mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right)}{\mathcal{N}_{\rm sim}}     \left( \mathcal{N}_{\rm sim} dm_B\,d x_1\, d_c \right)\, dz\, dm  \right]^N \\
        &= \left[ \frac{1}{N_{\rm sim}} \sum_{\rm passed} \frac{\mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right)}{\mathcal{N}_{\rm sim}}     \left( \mathcal{N}_{\rm sim} dm_B\,d x_1\, d_c \right)\, dz\, dm  \right]^N \\
        &=  \frac{1}{N_{\rm sim}^N} \left[\sum_{\rm passed} \frac{\mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right)}{\mathcal{N}_{\rm sim}}     \left( \mathcal{N}_{\rm sim} dm_B\,d x_1\, d_c \right)\, dz\, dm  \right]^N

    As the weights do not have to be normalised, we can discard the constant factor out front.

    .. math::
       w^N &\propto  \left[\sum_{\rm passed} \frac{\mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right)}{\mathcal{N}_{\rm sim}}     \left( \mathcal{N}_{\rm sim} dm_B\,d x_1\, d_c \right)\, dz\, dm  \right]^N \\
       \log\left(w^N\right) - {\rm const} &=  N \log\left[\sum_{\rm passed} \frac{\mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right)}{\mathcal{N}_{\rm sim}}     \left( \mathcal{N}_{\rm sim} dm_B\,d x_1\, d_c \right)\, dz\, dm  \right]

    Given a set of points to use in the integration, we can see that subtracting the above
    term from our likelihood provides a simple implementation of our bias correction.

.. warning::
    A primary concern with selection effects is that they grow exponentially worse with
    more data. To intuitively understand this, if you have an increased number of (biased)
    data points, the posterior maximum becomes better constrained and you need an increased
    re-weighting (bias correction) to shift the posterior maximum to the correct location.

    To provide a concrete example, suppose our weight (:math:`w`) is 0.99 in one section
    of the parameter space, and 1.01 in another section (normalised to some arbitrary point).
    With 300 data points, the difference in weights between those two points would be
    :math:`(1.01/0.99)^{300} \approx 404`. This difference in weights is potentially beyond
    the ability to re-weight an existing chain of results, and so the weights may need to
    be implemented directly inside the posterior evaluated by the fitting algorithm. We note
    that the 7th proof, :ref:`efficiency7`, shows undesired noise in the
    contours when looking at different values of :math:`\sigma`, and the ratio difference there
    for 2000 data points is only 81 (so 404 would be several times worse).

    An example of importance sampling an uncorrected posterior surface is shown below.

    .. figure::     ../dessn/models/d_simple_stan/output/plot_comparison.png
        :align:     center

        In blue we have the posterior surface for a likelihood that does not have any
        bias correction, and the red shows the same posterior after I have applied the
        :math:`w^-N` bias correction. Normalised to one, the mean weight of points
        after resampling is :math:`0.0002`, with the minimum weighted point weighted
        at :math:`2.7\times 10^{-13}`. The staggeringly low weights attributed
        is an artifact of the concerns stated above. The only good news I can see in this
        posterior is that there *does* seem to be a shift in :math:`\langle c \rangle` towards
        the correct value.

    If we focus on :math:`\langle c \rangle` for a second, we can see that the correct
    value falls roughly :math:`3\sigma` away from the sampled mean, and so this raises the
    question; *Is the issue simply that we have too few samples in the correct region of
    parameter space? Is that why our weights are on average so low?*

    The can investigate this easily. By looking at the selection efficiency simply as
    a function of apparent magnitude for the supernova simulated (that are used in the bias
    correction), we can implement an approximate *data dependent* bias correction using
    apparent magnitude (and colour) with an error function. That is identical to the
    current popular method of diving each supernova by its modelled selection efficiency,
    and so our likelihood becomes gains (for each supernova) a correcting term

    .. math::
        \Phi^{-1}(m_B | x_1, x_2) e^{\langle c \rangle},

    where :math:`\Phi` is the complimentary normal cumulative distribution function
    and :math:`x_1, x_2` respectively represent the mean apparent magnitude and the width
    of the cdf. It is important to note here that the actual functional form of the correction above
    does not matter - all we care about is that it moves the sampled region of the
    parameter space. We then apply the bias correction
    :math:`w^-N \Phi(m_B | x_1, x_2) e^{-\langle c \rangle}`, which implements our
    original bias correction whilst removing the approximate correction introduced
    to shift the region of sampling.

    .. figure::     ../dessn/models/d_simple_stan/output/approx_plot_full.png
        :align:     center

        In blue we have the posterior surface for a likelihood that does not have any
        bias correction, and the red shows the same posterior after I have applied the
        :math:`w^-N` bias correction. Normalised to one, the mean weight of points
        after resampling is :math:`0.001` (three times better than before). This is so
        far the most promising technique.


Given the concerns with the importance sampling methods, I also decided to implement
the bias corrections within STAN itself. Inserting the relevant data and structures
into STAn such that I can perform Monte Carlo integration in a BHM framework significantly
slows down the fits, however I believed it would at least give good results.

.. figure::     ../dessn/models/d_simple_stan/output/complete_plot_full.png
    :align:     center

    As you can see, I was wrong.

In addition to the odd contours, we can also see in the walk itself that we have
sampling issues, with some walkers sampling some areas of posterior space more than others.

.. figure::     ../dessn/models/d_simple_stan/output/complete_plot_walk.png
    :align:     center

Current Concerns
~~~~~~~~~~~~~~~~

1. Unsure how to fix STAN model with bias corrections inside. Cannot un-center those distributions more than they currently are.
2. Need to ensure that my results are not just "Got lucky with the random number seed". Will rerun fits realising multiple cosmologies.


"""