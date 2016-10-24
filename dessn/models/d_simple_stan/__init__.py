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
selection effects :math:`S`.
Our specific observations :math:`D` are the light curves themselves,
the summary statistics that result from them :math:`\lbrace \hat{m_B}, \hat{c}, \hat{x_1} \rbrace`,
the covariance for the summary statistics :math:`\hat{C}`, the redshifts of the
objects :math:`\hat{z}` and a normalised mass estimate :math:`\hat{m}`. We thus signify
observed variables with the hat operator. In this work we will be modelling
:math:`\lbrace \hat{m_B}, \hat{c}, \hat{x_1} \rbrace` as having true underlying
values, however assume that  :math:`\hat{z}` and :math:`\hat{m}` are
known :math:`(\hat{z} = z,\ \hat{m}=m)`.


.. math::
    :label: a

    P(\theta|D) &= \frac{ P(D|\theta) P(\theta) }{\int P(D|\theta^\prime) P(\theta^\prime)  \,d\theta^\prime}  \\[10pt]
                &\propto P(D|\theta) P(\theta) \\

As our likelihood :math:`\mathcal{L}` does not have a fixed normalisation constant,
but instead has :math:`\theta` dependent normalisation, we also must enforce normalisation of the likelihood:

.. math::
    :label: b

    \mathcal{L} &= \frac{ P(D|\theta)}{\int dD^\prime P(D^\prime|\theta)},

where :math:`D^\prime` represents all possible experimental outcomes. It is important to note that this is where
our selection effects kick in: :math:`D^\prime` is the subset of all possible data that satisfies our
selection effect, as only data that passes our cuts can be an experimental outcome that could
appear in the likelihood. We can formally model this inside the integral by denoting our selection function
as :math:`S(R)` that returns :math:`0` if the data does not pass the cut, and :math:`1` if the data does pass the cut,
such that

.. math::
    :label: c

    \int d D^\prime\  f(D^\prime) = \int d R\  S(R) f(R).

To put everything back together, we thus have

.. math::
    :label: cg

    P(\theta|D) &\propto \frac{P(D|\theta) P(\theta)}{\int d R\  S(R) P(D|\theta) }

----------

STAN Model
~~~~~~~~~~

Let us examine only the numerator for the time being. The numerator is the model
which ends up implemented in STAN, whilst the denominator can be implemented
differently. For simplicity, let us denote the population parameters
:math:`\lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle, \sigma_{M_B}, \sigma_{x_1}, \sigma_c, \rho \rbrace`
shown under the Population header as :math:`\gamma`.

Furthermore, in the interests of simplicity, let us examine only a single supernova for the time being. As I don't
yet take calibration into account, the supernova are independent. Let us denote the likelihood for a single
supernova as :math:`\mathcal{L}_i`.

.. math::
    :label: d

    \mathcal{L_i} P(\theta) &= P(\hat{m_B}, \hat{x_1}, \hat{c}, \hat{z}, \hat{m} |
    \Omega_m, w, \alpha, \beta, \gamma)
    P(\Omega_m, w, \alpha, \beta, \gamma) \\

Now, let us quickly deal with the priors so I don't have to type them out again and again.
We will treat :math:`\sigma_{M_B},\ \sigma_{x_1},\, \sigma_c`
with Cauchy priors, :math:`\rho` with an LKJ prior, and other parameters with flat priors.
So now we can focus on the likelihood's numerator, which is

.. math::
    :label: e

    \mathcal{L_i} &= P(\hat{m_B}, \hat{x_1}, \hat{c}, \hat{z}, \hat{m} |
    \Omega_m, w, \alpha, \beta, \gamma) \\[10pt]
    &= \int dm_B \int dx_1 \int dc \  P(\hat{m_B}, \hat{x_1}, \hat{c}, \hat{z}, \hat{m}, m_B, x_1, c | \Omega_m, w, \alpha, \beta, \gamma) \\[10pt]
    &= \int dm_B \int dx_1 \int dc \  P(\hat{m_B}, \hat{x_1}, \hat{c}, z, m, m_B, x_1, c | \Omega_m, w, \alpha, \beta, \gamma) \\[10pt]

Where in the last line I have used the fact that we assume mass and redshift are precisely known.
Also, as we assume that the observed summary statistics :math:`\hat{m_B}, \hat{x_1}, \hat{c}` are normally
distributed around the true values :math:`m_B,x_1,c`, we can separate them out:

.. math::
    :label: eg

    \mathcal{L_i} &= \int dm_B \int dx_1 \int dc \  P(\hat{m_B}, \hat{x_1}, \hat{c} | m_B, x_1, c, z, m, \Omega_m, w, \alpha, \beta, \gamma) P(m_B, x_1, c, z, m| \Omega_m, w, \alpha, \beta, \gamma) \\[10pt]
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

    P(D_i|\theta) P(\theta) &\propto
    \int dm_B \int dx_1 \int dc \int M_B\
    \rm{Cauchy}(\sigma_{M_B}|0,2.5)
    \rm{Cauchy}(\sigma_{x_1}|0,2.5)
    \rm{Cauchy}(\sigma_{c}|0,2.5)
    \rm{LKJ}(\rho|4) \\
    &\quad\quad\quad \mathcal{N}\left( \lbrace \hat{m_B}, \hat{x_1}, \hat{c} \rbrace | \lbrace m_B, x_1, c \rbrace, C \right)
    \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) \\
    &\quad\quad\quad \mathcal{N}\left( \lbrace M_B, x_1, c \rbrace |
    \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right) P(z) P(m)

We fit for this using 20 realisations of 200 supernova, is shown below. Note the bias in matter density
and mean colour (as the redder supernova are cut off at high redshift).


.. figure::     ../dessn/models/d_simple_stan/output/plot_simple_no_weight.png
    :align:     center

--------

Selection Effects
~~~~~~~~~~~~~~~~~

Having formulated a probabilistic model for the numerator of our posterior (and sent it off
to STAN), we can now turn our attention to the denominator :math:`w \equiv \int d R\  S(R) P(D|\theta)`.

As the bias correction is not data dependent, but model parameter dependent (cosmology dependent),
the correction for each data point is identical, such that the correction for each individual supernova
is identical.

We assume that selection effects can be determined as a function of apparent magnitude,
colour, stretch, redshift and mass.

.. math::
    :label: m

    w &= \int d\hat{m_B} \int d\hat{x}_1 \int d\hat{c}
    \int dz \int dm \int dm_B \int dx_1 \int dc \int dM_B\
    P(\hat{m_B}, m_B, \hat{x}_1, x_1, \hat{c}, c, z, m, M_B|\theta) S(m_B, x_1, c, z, m) \\[10pt]
    &= \idotsint d\hat{m_B}\, d\hat{x}_1 \, d\hat{c} \, dz \, dm \, dm_B \, dx_1 \, dc \, dM_B\
    \mathcal{N}\left( \lbrace \hat{m_B}, \hat{x_1}, \hat{c} \rbrace | \lbrace m_B, x_1, c \rbrace, C \right)\   S(m_B, x_1, c, z, m) \\
    &\quad\quad\quad  \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right)\
    \mathcal{N}\left( \lbrace M_B, x_1, c \rbrace |
    \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right) P(z) P(m)

Again that we assume redshift and mass are perfectly known, so the relationship between
actual (latent) redshift and mass and the observed quantity is a delta function, hence why
they only appear once in the equation above. The important assumption
is that the detection efficiency is to good approximation
captured by the apparent magnitude, colour, stretch, mass and redshift of the supernova.

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
    S(m_B, x_1, c, z, m) \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) P(M_B, x_1, c | \gamma) P(z) P(m) \\

Addressing each component individually:

.. math::
    :label: p

    P(z)&= \text{Redshift distribution from DES volume}\\
    P(m) &= \text{Unknown mass distribution} \\
    P(M_B, x_1, c|\gamma) &= \mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right) \\
    S(m_B, x_1, c, z, m) &= \text{If the data passes the cut} \\
    \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) &= \text{Transformation function}

Now enter the observational specifics of our survey: how many bands, the band passes,
frequency of observation, weather effects, etc. The selection effects we need to model are

    * At least 5 epochs between :math:`-99 < t < 60`.
    * :math:`0.0 < z < 1.2`.
    * At least one point :math:`t < -2`.
    * At least one point :math:`t > 10`.
    * At least 2 filters with :math:`S/N > 5`.

Finally, we note that, having :math:`N` supernova instead of one, we need only to normalise the likelihood
for each new point in parameter space, but not at each individual data point (supernova; because the normalisation
is independent of the data point). Thus our final posterior takes the following form, where I explicitly take into
account the number of supernova we have:

.. math::
    :label: final

    P(\theta|D) &\propto \frac{P(\theta)}{w^N} \idotsint d\vec{z}\,d\vec{m}\,d\vec{\hat{m_B}}\, d\vec{m_B}\,
    d\vec{\hat{x_1}}\,  d\vec{x_1}\, d\vec{\hat{c}}\, d\vec{c}
    \prod_{i=1}^N P(D_i|\theta)





.. note::
    :class: green

    **Technical aside**: Calculating :math:`S(m_B, x_1, c, z, m)`
    is not an analytic task. It has complications not just in the distance modulus being the
    result of an integral, but also that the colour and stretch correction factors make
    extra use of supernova specific values. The way to efficiently determine the efficiency
    is given as follows:

        1. Initially run a large DES-like simulation, recording all generated SN parameters and whether they pass the cuts.
        2. Using input cosmology to translate :math:`m_B, x_1, c` distribution to a :math:`M_B, x_1, c` distribution.
        3. Perform Monte-Carlo integration using the distribution.

    To go into the math, our Monte Carlo integration for the weights. Our initial sample
    of supernova simulated is drawn from the multivariate normal distribution :math:`\mathcal{N}_{\rm sim}`.

    .. math::
        w^N &= \left[ \frac{1}{N_{\rm sim}} \sum  P(S|m_B, x_1, c, z,m)  \frac{\mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right)}{\mathcal{N}_{\rm sim}}     \left( \mathcal{N}_{\rm sim} dm_B\,d x_1\, d_c \right)\, dz\, dm  \right]^N \\
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

    .. figure::     ../dessn/models/d_simple_stan/output/plot_simple_single_weight.png
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

    .. figure::     ../dessn/models/d_simple_stan/output/plot_approx_weight.png
        :align:     center

        In blue we have the posterior surface for a likelihood that does not have any
        bias correction, and the red shows the same posterior after I have applied the
        :math:`w^{-N}` bias correction. Normalised to one, the mean weight of points
        after resampling is :math:`0.001` (three times better than before). This is so
        far the most promising technique.


Given the concerns with the importance sampling methods, I also decided to implement
the bias corrections within STAN itself. Inserting the relevant data and structures
into STAn such that I can perform Monte Carlo integration in a BHM framework significantly
slows down the fits, however I believed it would at least give good results.

.. figure::     ../dessn/models/d_simple_stan/output/plot_stan_mc_single.png
    :align:     center

    As you can see, I was wrong.

In addition to the odd contours, we can also see in the walk itself that we have
sampling issues, with some walkers sampling some areas of posterior space more than others.

.. figure::     ../dessn/models/d_simple_stan/output/plot_stan_mc_walk.png
    :align:     center


"""