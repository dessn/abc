r""" My attempt at a proper STAN model.

**Note to Alex/Tam:** Currently have turned off mass and calibration to try and get the
approximate correction working.


I follow Rubin et al. (2015) with some changes.

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

**Cosmological parameters**:

    * :math:`\Omega_m`: matter density
    * :math:`w`: dark energy equation of state
    * :math:`\alpha`: Phillips correction for stretch
    * :math:`\beta`: Tripp correction for colour

**Population parameters**:

    * :math:`\langle M_B \rangle`: mean absolute magnitude of supernova
    * :math:`\sigma_{M_B}`: standard deviation of absolute magnitudes
    * :math:`\langle c \rangle`: mean colour
    * :math:`\sigma_c`: standard deviation of  colour
    * :math:`\langle x_1 \rangle`: mean scale
    * :math:`\sigma_{x_1}`: standard deviation of scale
    * :math:`\rho`: correlation (matrix) between absolute magnitude, colour and stretch

**Marginalised parameters**:
    * :math:`\delta(0)` and :math:`\delta(\infty)`: The magnitude-mass relationship
    * :math:`\delta \mathcal{Z}_b`: Zeropoint uncertainty for each of the *g,r,i,z* bands.

**Per supernova parameters**:

    * :math:`m_B`: the true (latent) apparent magnitude
    * :math:`x_1`: the true (latent) stretch
    * :math:`c`: the true (latent) colour
    * :math:`z`: the true redshift of the supernova
    * :math:`m`: the true mass of the host galaxy

----------

Model Overview
--------------

We wish to model our posterior, given our observations, our model :math:`\theta`, and
selection effects :math:`S`.
Our specific observations :math:`D` are the light curves themselves,
the summary statistics that result from them :math:`\lbrace \hat{m_B}, \hat{c}, \hat{x_1} \rbrace`,
the covariance for the summary statistics :math:`\hat{C}`, the redshifts of the
object :math:`\hat{z}` and a normalised mass estimate :math:`\hat{m}`. We thus signify
observed variables with the hat operator. In this work we will be modelling
:math:`\lbrace \hat{m_B}, \hat{c}, \hat{x_1} \rbrace` as having true underlying
values, however assume that  :math:`\hat{z}` and :math:`\hat{m}` are
known :math:`(\hat{z} = z,\ \hat{m}=m)`.

For simplicity, we adopt the commonly used notation that :math:`\eta\equiv \lbrace \hat{m_B}, \hat{c}, \hat{x_1} \rbrace`.


.. math::
    :label: a

    P(\theta|D) &= \frac{ \mathcal{L}(D|\theta) P(\theta) }{\int \mathcal{L}(D|\theta^\prime) P(\theta^\prime)  \,d\theta^\prime}  \\[10pt]
                &\propto \mathcal{L}(D|\theta) P(\theta) \\

with

.. math::
    :label: b

    \mathcal{L}(D|\theta) &= \frac{ P(D|\theta)}{\int  P(D^\prime|\theta) \ dD^\prime}, \\

where :math:`D^\prime` represents all possible experimental outcomes. It is important to note that this is where
our selection effects kick in: :math:`D^\prime` is the subset of all possible data that satisfies our
selection effect, as only data that passes our cuts can be an experimental outcome that could
appear in the likelihood. We can formally model this inside the integral by denoting our selection function
as :math:`S(R)` that returns :math:`0` if the data :math:`R` does not pass the cut, and :math:`1` if the input data does
pass the cut, such that

.. math::
    :label: c

    \int d D^\prime\  f(D^\prime) = \int d R\  S(R) f(R),\\

where :math:`R` represents all possible data - not just all possible experimental outcomes.
To put everything back together, we thus have

.. math::
    :label: cg

    P(\theta|D) &\propto \frac{P(D|\theta) P(\theta)}{\int  S(R) P(R|\theta) \ dR  } \\[10pt]
    \equiv frac{P(D|\theta) P(\theta)}{w  }

where in the last line to make the notation easier I define :math:`w \equiv \int  S(R) P(R|\theta) \ dR`.

An equivalent way of writing this down is rephrase the probability of attaining an experimental outcome
as the product of an event occurring given :math:`\theta` multiplied by the conditional probability
of our experiment observing the event, given that it did occur. I simply write out :math:`S(R)` instead
of writing it as a conditional probability to reduce the number of subscripts in the notation.

----------

STAN Model
----------

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

    \mathcal{P_i} (D|\theta) P(\theta) &= P(\hat{m_B}, \hat{x_1}, \hat{c}, \hat{z}, \hat{m} |
    \Omega_m, w, \alpha, \beta, \gamma, \delta \mathcal{Z}_b, z, m)
    P(\Omega_m, w, \alpha, \beta, \gamma, \delta \mathcal{Z}_b, z, m) \\

Now, let us quickly deal with the priors so I don't have to type them out again and again.
We will treat :math:`\sigma_{M_B},\ \sigma_{x_1},\, \sigma_c`
with Cauchy priors, :math:`\rho` with an LKJ prior, :math:`\delta \mathcal{Z}_b` is constrained by
zero point uncertainty from photometry (currently just set to 0.01 mag normal uncertainty)
and other parameters with flat priors. The prior
distributions on redshift and host mass do not matter in this likelihood (without bias corrections),
as we assume redshift and mass are precisely known.
So now we can focus on the likelihood's numerator, which is

.. math::
    :label: e

    \mathcal{L_i} &= \iiint d\eta \  P(\hat{\eta}, \eta |  z, m, \Omega_m, w, \alpha, \beta, \gamma, \delta \mathcal{Z}_b )  \delta(\hat{z} - z) \delta(\hat{m}-m) \\[10pt]

.. admonition:: Show/Hide derivation
   :class: toggle note math

    .. math::

        \mathcal{L_i} &= P(\hat{m_B}, \hat{x_1}, \hat{c}, \hat{z}, \hat{m} |
        \Omega_m, w, \alpha, \beta, \gamma, \delta \mathcal{Z}_b, z, m) \\[10pt]
        &= \int dm_B \int dx_1 \int dc \  P(\hat{m_B}, \hat{x_1}, \hat{c}, \hat{z}, \hat{m}, m_B, x_1, c | \Omega_m, w, \alpha, \beta, \gamma, \delta \mathcal{Z}_b, z, m) \\[10pt]
        &= \iiint d\eta \  P(\hat{\eta}, \hat{z}, \hat{m}, \eta | \Omega_m, w, \alpha, \beta, \gamma, \delta \mathcal{Z}_b, z, m) \\[10pt]
        &= \iiint d\eta \  \delta(\hat{z} - z) \delta(\hat{m}-m) P(\hat{\eta}, z, m, \eta | \Omega_m, w, \alpha, \beta, \gamma, \delta \mathcal{Z}_b, z, m) \\[10pt]
        &= \iiint d\eta \  P(\hat{\eta}, \eta |  z, m, \Omega_m, w, \alpha, \beta, \gamma, \delta \mathcal{Z}_b )  \delta(\hat{z} - z) \delta(\hat{m}-m) \\[10pt]

Where I have used the fact that we assume mass and redshift are precisely known
(:math:`\hat{z}=z` and :math:`\hat{m}=m`), and therefore do not need to be modelled with latent parameters.
We take zeropoint uncertainty into account by computing :math:`\frac{\partial\hat{\eta}}{\partial\mathcal{Z}_b}` for each supernova
light curve. We thus model what would be the observed values :math:`\hat{\eta}_{\rm True} = \hat{\eta} + \delta\mathcal{Z}_b \frac{\partial\hat{\eta}}{\partial\mathcal{Z}_b}`,
and then assume that true observed summary statistics :math:`\hat{\eta}_{\rm True}` are normally
distributed around the true values :math:`\eta`, we can separate them out.

.. math::
    :label: eg

    \mathcal{L_i} &= \iiint d\eta \  \mathcal{N}\left( \hat{\eta} + \delta\mathcal{Z}_b \frac{\partial\hat{\eta}}{\partial\mathcal{Z}_b} |\eta, C \right) P(\eta| z, m, \Omega_m, w, \alpha, \beta, \gamma)  \delta(\hat{z} - z) \delta(\hat{m}-m)  \\


.. admonition:: Show/Hide derivation
   :class: toggle note math

    .. math::

        \mathcal{L_i} &= \iiint d\eta \  P(\hat{\eta} | \eta, z, m, \Omega_m, w, \alpha, \beta, \gamma, \delta \mathcal{Z}_b ) P(\eta| z, m, \Omega_m, w, \alpha, \beta, \gamma, \delta \mathcal{Z}_b ) \delta(\hat{z} - z) \delta(\hat{m}-m)  \\[10pt]
        &= \iiint d\eta \  P(\hat{\eta} | \eta, \delta \mathcal{Z}_b) P(\eta | z, m, \Omega_m, w, \alpha, \beta, \gamma)  \delta(\hat{z} - z) \delta(\hat{m}-m)  \\[10pt]
        &= \iiint d\eta \  \mathcal{N}\left( \hat{\eta} + \delta\mathcal{Z}_b \frac{\partial\hat{\eta}}{\partial\mathcal{Z}_b} |\eta, C \right) P(\eta| z, m, \Omega_m, w, \alpha, \beta, \gamma)  \delta(\hat{z} - z) \delta(\hat{m}-m)  \\

Now, in order to calculate :math:`P(\eta| \Omega_m, w, \alpha, \beta, \gamma, z, m, \delta\mathcal{Z}_b)`,
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

We note that :math:`\mu` is a function of :math:`\hat{z},\Omega_m,w`, however we will simply denote it
:math:`mu` to keep the notation from spreading over too many lines.

From the above,  :math:`M_B` is a function of :math:`\Omega_m, w, \alpha, \beta, x_1, c, z, m`. Or, more probabilistically,

.. math::
    P(M_B, m_B) = \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right).

We can thus introduce a latent variable :math:`M_B` and immediately remove the :math:`m_B` integral via the delta function.

.. math::
    :label: i

    \mathcal{L} &= \iiint d\eta  \int M_B \  \mathcal{N}\left( \hat{\eta} + \delta\mathcal{Z}_b \frac{\partial\hat{\eta}}{\partial\mathcal{Z}_b} | \eta, C \right) P(\eta, M_B | z, m, \Omega_m, w, \alpha, \beta, \gamma, \delta\mathcal{Z}_b) \delta(\hat{z} - z) \delta(\hat{m}-m) \\[10pt]

where

.. math::
    :label: igg

    P(\eta, M_B| \theta) &= \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) \mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right) \delta(\hat{z} - z) \delta(\hat{m}-m) \\[10pt]

.. admonition:: Show/Hide derivation
   :class: toggle note math

    .. math::
        :label: ig

        P(\eta, M_B| \theta) &= P(m_B | M_B, x_1, c, z, m, \Omega_m, w, \alpha, \beta, \gamma, \delta\mathcal{Z}_b ) P (M_B, x_1, c, | z, m, \Omega_m, w, \alpha, \beta, \gamma, \delta\mathcal{Z}_b )\delta(\hat{z} - z) \delta(\hat{m}-m) \\[10pt]
        &= \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) P (M_B, x_1, c | z, m,\Omega_m, w, \alpha, \beta, \gamma, \delta\mathcal{Z}_b ) \delta(\hat{z} - z) \delta(\hat{m}-m) \\[10pt]
        &= \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) P (M_B, x_1, c, | \gamma) \delta(\hat{z} - z) \delta(\hat{m}-m)\\[10pt]
        &= \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) \mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right) \delta(\hat{z} - z) \delta(\hat{m}-m) \\[10pt]

with

.. math::
    :label: j

    V &= \begin{pmatrix}
    \sigma_{M_B}^2                        & \rho_{12} \sigma_{M_B} \sigma_{x_1}         & \rho_{13} \sigma_{M_B} \sigma_{c}  \\
    \rho_{21} \sigma_{M_B} \sigma_{x_1}           & \sigma_{x_1}^2                    & \rho_{23} \sigma_{x_1} \sigma_{c}  \\
    \rho_{31} \sigma_{M_B} \sigma_{c}          & \rho_{32} \sigma_{x_1} \sigma_{c}       & \sigma_{c}^2  \\
    \end{pmatrix}

giving the population covariance.


.. note::
    In this implementation there is no skewness in the colour distribution.
    As we do not require normalised probabilities, we can simply add in correcting
    factors that can emulate skewness. This has been done in the ``simple_skew`` model, where we
    add in a CDF probability for the colour to turn our normal into a skew normal.

Putting this back together, we now have a simple hierarchical multi-normal model.
Adding in the priors, and taking into account that we observe multiple supernova, we have
that a final numerator of:



.. math::
    :label: k

    P(D_i|\theta) P(\theta) &\propto
    \rm{Cauchy}(\sigma_{M_B}|0,2.5)
    \rm{Cauchy}(\sigma_{x_1}|0,2.5)
    \rm{Cauchy}(\sigma_{c}|0,2.5)
    \rm{LKJ}(\rho|4) \\
    &\quad  \iiint d\eta_i \int M_{Bi}\
    \mathcal{N}\left( \hat{\eta_i} + \delta\mathcal{Z}_b \frac{\partial\hat{\eta_i}}{\partial\mathcal{Z}_b} | \eta_i, C_i \right)
    \delta\left(M_{Bi} - \left[ m_{Bi} - \mu_i + \alpha x_{1i} - \beta c_i + k(z_i) m_i\right]\right) \\
    &\quad\quad\quad \mathcal{N}\left( \lbrace M_{Bi}, x_{1i}, c_i \rbrace |
    \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right) \delta(\hat{z_i} - z_i) \delta(\hat{m_i}-m_i)

--------

Selection Effects
-----------------

Now, the easy part of the model is done, we need to move on to the real issue - our data is biased.
As the bias correction is not data dependent, but model parameter dependent (cosmology dependent),
the correction for each data point is identical, such that the correction for each individual supernova
is identical.

We assume, for any given supernova, the selection effect can be determined as a function of apparent magnitude,
colour, stretch, redshift and mass. We might expect that the zero points have an effect
on selection efficiency, however this is because we normally consider zero points and
photon counts hand in hand. As we have a fixed experiment (fixed photon counts and statistics)
with different zero points, the selection efficiency is actually independent from zero points. Thus, we can
compute the bias correction as

.. math::
    :label: mmm

    w &= \idotsint d\hat{\eta} \, d\eta \, dz\, dm\, dM_B\
    \mathcal{N}\left( \hat{\eta} + \delta\mathcal{Z}_b \frac{\partial\hat{\eta}}{\partial\mathcal{Z}} | \eta, C \right)\   S(m_B, x_1, c, z, m) \\
    &\quad\quad\quad  \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right)\
    \mathcal{N}\left( \lbrace M_B, x_1, c \rbrace |
    \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right) \\

.. admonition:: Show/Hide derivation
   :class: toggle note math

    .. math::
        :label: m

        w &= \iiint d\hat{\eta} \iiint d\eta \int dM_B\  \int d\hat{z} \int \hat{m} \int dz \int dm \,
        P(\hat{\eta},\eta, \hat{z},z, \hat{m},m, M_B|\theta) S(m_B, x_1, c, z, m) \\[10pt]
        &= \idotsint d\hat{\eta} \, d\eta \, d\hat{z} \, dz\, d\hat{m}\, dm\, dM_B\
        \mathcal{N}\left( \hat{\eta} + \delta\mathcal{Z}_b \frac{\partial\hat{\eta}}{\partial\mathcal{Z}} | \eta, C \right)\   S(m_B, x_1, c, z, m) \\
        &\quad\quad\quad  \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right)\
        \mathcal{N}\left( \lbrace M_B, x_1, c \rbrace |
        \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right)\delta(\hat{z} - z) \delta(\hat{m}-m) \\[10pt]
        &= \idotsint d\hat{\eta} \, d\eta \, dz\, dm\, dM_B\
        \mathcal{N}\left( \hat{\eta} + \delta\mathcal{Z}_b \frac{\partial\hat{\eta}}{\partial\mathcal{Z}} | \eta, C \right)\   S(m_B, x_1, c, z, m) \\
        &\quad\quad\quad  \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right)\
        \mathcal{N}\left( \lbrace M_B, x_1, c \rbrace |
        \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right) \\

Again that we assume redshift and mass are perfectly known, so the relationship between
actual (latent) redshift and mass and the observed quantity is a delta function, hence why
they only appear once in the equation above. The important assumption
is that the detection efficiency is to good approximation
captured by the apparent magnitude, colour, stretch, mass and redshift of the supernova.

As we integrate over all possible realisations, we have that over all space

.. math::
    :label: n

    \iiint d\hat{\eta} \, P(\hat{\eta} | \eta, \delta\mathcal{Z}_b) =
    \iiint_{-\infty}^{\infty} d\hat{\eta}\,
    \mathcal{N}\left( \hat{\eta} + \delta\mathcal{Z}_b \frac{\partial\hat{\eta}}{\partial\mathcal{Z}} | \eta, C \right) = 1 \\

and as such we can remove it from the integral. As is expected, the final weight looks exactly like our likelihood,
except with some extra integral signs that marginalise over all possible experimental realisations:

.. math::
    :label: o

    w &= \idotsint d\eta\, dz \, dm \, dM_B\
    S(m_B, x_1, c, z, m) \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) P(M_B, x_1, c | \gamma) \\

Addressing each component individually:

.. math::
    :label: p

    P(M_B, x_1, c|\gamma) &= \mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right) \\
    S(m_B, x_1, c, z, m) &= \text{If the supernova (light curves and summary stats) pass the cut} \\
    \delta\left(M_B - \left[ m_B - \mu + \alpha x_1 - \beta c + k(z) m\right]\right) &= \text{Transformation function} \\

Finally, we note that, having :math:`N` supernova instead of one, we need only to normalise the likelihood
for each new point in parameter space, but not at each individual data point (because the normalisation
is independent of the data point). Thus our final posterior takes the following form, where I explicitly take into
account the number of supernova we have:

.. math::
    :label: final

    P(\theta|D) &\propto \frac{P(\theta)}{w^N} \idotsint d\vec{m_B}\, d\vec{x_1}\, \, d\vec{c}\, d\vec{M_B} \prod_{i=1}^N P(D_i|\theta) \\





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

    This gives our correction :math:`w` as

    .. math::
        :label: techw1

        w \propto \left[\sum_{\rm passed} \frac{\mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right)}{\mathcal{N}_{\rm sim}}     \left( \mathcal{N}_{\rm sim} dm_B\,d x_1\, d_c \right)\, dz\, dm  \right]^N \\

    .. admonition:: Show/Hide derivation
        :class: toggle note math

        To go into the math, our Monte Carlo integration sample
        of simulated supernova is drawn from the multivariate normal distribution :math:`\mathcal{N}_{\rm sim}`.

        .. math::
            :label: techw2

            w^N &= \left[ \frac{1}{N_{\rm sim}} \sum  P(S|m_B, x_1, c, z,m)  \frac{\mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right)}{\mathcal{N}_{\rm sim}}     \left( \mathcal{N}_{\rm sim} dm_B\,d x_1\, d_c \right)\, dz\, dm  \right]^N \\
            &= \left[ \frac{1}{N_{\rm sim}} \sum_{\rm passed} \frac{\mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right)}{\mathcal{N}_{\rm sim}}     \left( \mathcal{N}_{\rm sim} dm_B\,d x_1\, d_c \right)\, dz\, dm  \right]^N \\
            &=  \frac{1}{N_{\rm sim}^N} \left[\sum_{\rm passed} \frac{\mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right)}{\mathcal{N}_{\rm sim}}     \left( \mathcal{N}_{\rm sim} dm_B\,d x_1\, d_c \right)\, dz\, dm  \right]^N

        As the weights do not have to be normalised, we can discard the constant factor out front. We also note that
        determining whether a simulated supernova has passed the cut now means converting light curve counts to flux
        and checking that the new fluxes pass signal-to-noise cuts.

        .. math::
            :label: techw3

            w^N &\propto  \left[\sum_{\rm passed} \frac{\mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right)}{\mathcal{N}_{\rm sim}}     \left( \mathcal{N}_{\rm sim} dm_B\,d x_1\, d_c \right)\, dz\, dm  \right]^N \\
            \log\left(w^N\right) - {\rm const} &=  N \log\left[\sum_{\rm passed} \frac{\mathcal{N}\left( \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right)}{\mathcal{N}_{\rm sim}}     \left( \mathcal{N}_{\rm sim} dm_B\,d x_1\, d_c \right)\, dz\, dm  \right]

        Given a set of points to use in the integration, we can see that subtracting the above
        term from our log-likelihood provides an implementation of our bias correction.

.. warning::
    A primary concern with selection effects is that they grow exponentially worse with
    more data. To intuitively understand this, if you have an increased number of (biased)
    data points, the posterior maximum becomes better constrained and you need an increased
    re-weighting (bias correction) to shift the posterior maximum to the correct location. Because
    of this, we will need to implement an approximate bias correction in Stan.

    .. admonition:: Show/Hide discussion
        :class: toggle note math

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
            :width: 70%

            In blue we have the posterior surface for a likelihood that does not have any
            bias correction, and the red shows the same posterior after I have applied the
            :math:`w^{-N}` bias correction. Normalised to one, the mean weight of points
            after resampling is :math:`0.0002`, with the minimum weighted point weighted
            at :math:`2.7\times 10^{-13}`. The staggeringly low weights attributed
            is an artifact of the concerns stated above. The only good news I can see in this
            posterior is that there *does* seem to be a shift in :math:`\langle c \rangle` towards
            the correct value.

        If we focus on :math:`\langle c \rangle` for a second, we can see that the correct
        value falls roughly :math:`3\sigma` away from the sampled mean, and so this raises the
        question; *Is the issue simply that we have too few samples in the correct region of
        parameter space? Is that why our weights are on average so low?*

To recap, we have a full bias correction that can be computed using Monte-Carlo integration. However,
Monte-Carlo integration cannot be put inside the Stan framework, and having no bias corretion
at all in the Stan framework means that our sampling efficiency drops to close to zero, which makes
it very difficult to adequately sample the posterior adequately. As such, we need an approximate
bias correction which *can* go inside Stan to improve our efficiency.

We can do this by looking at the selection efficiency simply as
a function of apparent magnitude for the supernova. There are two possibilities that we can do. The first
is to approximate the selection efficiency as a normal CDF, as was done in Rubin (2005). However, when
simulating the DES data, low spectroscopic efficiency at brighter magnitudes makes a CDF an inappropriate
choice. Instead, the most general analytic form we can prescribe the approximate correction
would be using a skew normal, as (depending on the value of the skew parameter :math:`\alpha`) we
can smoothly transition from a normal CDF to a normal PDF. Thus the approximate bias function
is described by

.. math::
    :label: approxbiassmall

    w_{\rm approx} &= \int dz \left[ \int dm_B \,  S(m_B) P(m_B|z,\theta) \right] P(z|\theta) \\[10pt]

.. admonition:: Show/Hide derivation
   :class: toggle note math

    .. math::
        :label: approxbias

        w_{\rm approx} &= \int d\hat{z} \int d\hat{m_B} \, P(\hat{z},\hat{m_B}|\theta) S(m_B) \\[10pt]
        &= \int d\hat{z} \int d\hat{m_B} \int dz \int dm_B \, P(\hat{z},\hat{m_B},z,m_B|\theta)  S(m_B) \\[10pt]
        &= \int d\hat{z} \int d\hat{m_B} \int dz \int dm_B \, P(\hat{z}|z) P(\hat{m_B}|m_B) P(m_B|z,\theta) P(z|\theta)  S(m_B) \\[10pt]
        &= \int d\hat{z} \int d\hat{m_B} \int dz \int dm_B \, \delta(\hat{z}-z) \mathcal{N}(\hat{m_B}|m_B,\hat{\sigma_{m_B}}) P(m_B|z,\theta) P(z|\theta)  S(m_B) \\[10pt]
        &= \int dz \int dm_B \, \left[ \int d\hat{m_B} \mathcal{N}(\hat{m_B}|m_B,\hat{\sigma_{m_B}}) \right] P(m_B|z,\theta) P(z|\theta) S(m_B) \\[10pt]
        &= \int dz \left[ \int dm_B \,  S(m_B) P(m_B|z,\theta) \right] P(z|\theta) \\[10pt]



As such, we have our efficiency function

.. math::
    :label: ee1

    S(m_B) = \mathcal{N}_{\rm skew} (m_B | m_{B,{\rm eff}}, \sigma_{{\rm eff}}, \alpha_{\rm eff})\\

With our survey efficiency thus defined, we need to describe our supernova model as a population
in apparent magnitude (and redshift). This will be given by a normal function with mean
:math:`m_B^*(z) = \langle M_B \rangle + \mu(z) - \alpha \langle x_1 \rangle + \beta \langle c \rangle`.
The width of this normal is then given by
:math:`(\sigma_{m_B}^*)^2 = \sigma_{m_B}^2 + (\alpha \sigma_{x_1})^2 + (\beta \sigma_c)^2 + 2(\alpha \sigma_{m_B,x_1} + \beta \sigma_{m_B,c} + \alpha\beta\sigma_{x_1,c})`,
such that we formally have

.. math::
    :label: poppdf

    P(m_B | z,\theta) &= \mathcal{N}(m_B | m_B^*(z), \sigma_{m_B}^*) \\

From this, we can derive an approximate weight :math:`w^*`:

.. math::
    :label: wstarshort

    w_{\rm approx} &= 2 \int dz \,
    \mathcal{N} \left( \frac{ m_{B,{\rm eff}} - m_B^*(z) }{ \sqrt{ \sigma_{{\rm eff}}^2 + \sigma_{m_B}^{*2} }} \right)
    \Phi\left( \frac{m_B^*(z) - m_{B,{\rm eff}} }{ \frac{\sigma_{m_B}^{*2} +  \sigma_{{\rm eff}}^2}{\sigma_{{\rm eff}}^2} \sqrt{ \left( \frac{ \sigma_{{\rm eff}} }{ \alpha_{\rm eff} }  \right)^2 +      \frac{  \sigma_{m_B}^{*2} \sigma_{{\rm eff}}^2  }{ \sigma_{m_B}^{*2} +  \sigma_{{\rm eff}}^2 }        } }  \right)
    P(z|\theta) \\[10pt]


.. admonition:: Show/Hide derivation
   :class: toggle note math

    .. math::
        :label: wstar

        w_{\rm approx} &= \int dz \left[ \int dm_B \,  S(m_B) P(m_B|z,\theta) \right] P(z|\theta) \\[10pt]
        &= \int dz \left[
        \int dm_B \,  \mathcal{N}_{\rm skew} (m_B | m_{B,{\rm eff}}, \sigma_{{\rm eff}}, \alpha_{\rm eff})
        \mathcal{N}(m_B | m_B^*(z), \sigma_{m_B}^*)
        \right] P(z|\theta) \\[10pt]
        &= 2 \int dz \left[
        \int dm_B \,  \mathcal{N} \left(\frac{m_B - m_{B,{\rm eff}}}{\sigma_{{\rm eff}}}\right) \Phi\left(\alpha_{\rm eff} \frac{m_B - m_{B,{\rm eff}}}{\sigma_{{\rm eff}}}\right)
        \mathcal{N}\left(\frac{m_B - m_B^*(z)}{\sigma_{m_B}^*}\right)
        \right] P(z|\theta) \\[10pt]
        &= 2 \int dz \left[ \int dm_B \,
        \mathcal{N} \left( \frac{ m_{B,{\rm eff}} - m_B^*(z) }{ \sqrt{ \sigma_{{\rm eff}}^2 + \sigma_{m_B}^{*2} }} \right)
        \mathcal{N} \left( \frac{ m_B - \bar{m_B} }{  \bar{\sigma}_{m_B}  }\right)
        \Phi\left(\alpha_{\rm eff} \frac{m_B - m_{B,{\rm eff}}}{\sigma_{{\rm eff}}}\right)
        \right] P(z|\theta) \\[10pt]
        & {\rm where }\ \  \bar{m_B} = \left( m_{B,{\rm eff}} \sigma_{m_B}^{*2} +   m_B^*(z) \sigma_{{\rm eff}}^2 \right) / \left( \sigma_{m_B}^{*2} +  \sigma_{{\rm eff}}^2 \right)  \\[10pt]
        & {\rm where }\ \  \bar{\sigma}_{m_B}^2 = \left(  \sigma_{m_B}^{*2} \sigma_{{\rm eff}}^2  \right) / \left( \sigma_{m_B}^{*2} +  \sigma_{{\rm eff}}^2 \right)   \\[10pt]
        &= 2 \int dz \,
        \mathcal{N} \left( \frac{ m_{B,{\rm eff}} - m_B^*(z) }{ \sqrt{ \sigma_{{\rm eff}}^2 + \sigma_{m_B}^{*2} }} \right)
        \left[ \int dm_B \,
        \mathcal{N} \left( \frac{ m_B - \bar{m_B} }{  \bar{\sigma}_{m_B}  }\right)
        \Phi\left(\alpha_{\rm eff} \frac{m_B - m_{B,{\rm eff}}}{\sigma_{{\rm eff}}}\right)
        \right] P(z|\theta) \\[10pt]
        &= 2 \int dz \,
        \mathcal{N} \left( \frac{ m_{B,{\rm eff}} - m_B^*(z) }{ \sqrt{ \sigma_{{\rm eff}}^2 + \sigma_{m_B}^{*2} }} \right)
        \Phi\left( \frac{\bar{m_B} - m_{B,{\rm eff}} }{ \sqrt{ \left( \frac{ \sigma_{{\rm eff}} }{ \alpha_{\rm eff} }  \right)^2 + 2\bar{\sigma}_{m_B}^2 } }  \right)
        P(z|\theta) \\[10pt]
        &= 2 \int dz \,
        \mathcal{N} \left( \frac{ m_{B,{\rm eff}} - m_B^*(z) }{ \sqrt{ \sigma_{{\rm eff}}^2 + \sigma_{m_B}^{*2} }} \right)
        \Phi\left( \frac{m_B^*(z) - m_{B,{\rm eff}} }{ \frac{\sigma_{m_B}^{*2} +  \sigma_{{\rm eff}}^2}{\sigma_{{\rm eff}}^2} \sqrt{ \left( \frac{ \sigma_{{\rm eff}} }{ \alpha_{\rm eff} }  \right)^2 +      2\frac{  \sigma_{m_B}^{*2} \sigma_{{\rm eff}}^2  }{ \sigma_{m_B}^{*2} +  \sigma_{{\rm eff}}^2 }        } }  \right)
        P(z|\theta) \\[10pt]


    `Thank you Wikipedia for laying out the second last line out so nicely <https://en.wikipedia.org/wiki/Error_function#Integral_of_error_function_with_Gaussian_density_function>`_.

We can see here that as our skew normal approaches a normal (:math:`\alpha \rightarrow 0`), the CDF function tends to
:math:`\frac{1}{2}` and gives us only the expected normal residual.


.. note::

    If we wanted to use the original complimentary CDF approximation for the selection efficiency, we would get the integral
    of the complimentary CDF function.

    .. math::
        :label: wstarshort2

        w_{\rm approx} &= 2 \int dz \,
        \Phi^c\left( \frac{m_B^* - m_{B,{\rm eff}}}{\sqrt{ {\sigma_{m_B}^*}^2 +   \sigma_{{\rm survey}}^2}} \right)
        P(z|\theta) \\[10pt]

    Now here we depart from Rubin (2015). Rubin (2015) formulate their likelihood in terms of a combinatorial
    problem, taking into account the number of observed events and an unknown number of missed events. Detailed in
    their Appendix B, they also make "the counterintuitive approximation that the redshift of each missed
    SN is exactly equal to the redshift of a detected SN. This approximation is accurate because the SN samples have,
    on average, enough SNe that the redshift distribution is reasonable sampled."

    **Alex/Tam, looking for feedback:** Unfortunately, I must disagree that this approximation is valid, because
    whilst the SN surveys *may* be able to reasonable sample the *observed* redshift distribution of SN, they
    *do not* adequately sample the underlying redshift distribution. Now, the underlying redshift distribution
    goes to a very high redshift, however we note that we would not have to integrate over all of it, because
    above the observed redshift distribution the contribution to the integral quckly drops to zero. However,
    sampling the high redshift tail is still necessary.

    It is of interest that the difference in methodology (between my integral and Rubin's
    combinatorics/redshift approximation/geometric series leads to the following difference in bias corrections.

    Note that I use capital W below, to denote a correction for the entire model, not a single supernova.

    .. math::
        W_{\rm approx} &= 2 \left(\int dz \,
        \Phi^c\left( \frac{m_B^* - m_{B,{\rm eff}}}{\sqrt{ {\sigma_{m_B}^*}^2 +   \sigma_{{\rm survey}}^2}} \right)
        P(z|\theta) \right)^N \\[10pt]
        W_{\rm Rubin} &= \prod_{i=1}^N \frac{P({\rm detect}|\lbrace \hat{m_{Bi}}, \hat{x_{1i}}, \hat{c_i} \rbrace) }{P({\rm detect} | z_i) }
        = \prod_{i=1}^N (\epsilon + P({\rm detect} | z_i))^{-1} \\

    where the last line utilises a small :math:`\epsilon` to aid convergences, and we discard the
    numerator as Rubin states with :math:`\epsilon > 0` it didn't turn out to be important.

    To try and compare these different methods, I've also tried a similar exact redshift approximation
    to reduce my integral down to a product, however it does not work well. That said, nothing I have
    tried has worked well, so maybe it is, in fact, alright.


After fitting the above posterior surface, we can remove the approximation correction
and implement the actual Monte Carlo correction by assigning each point the chain the weight based on the
difference between the approximate weight and the actual weight.


Final Model
-----------

To lay out all the math in one go, the blue section represents the model fitted using STAN, and the red math
represents are post-fit weight corrections to correctly take into account bias.

.. math::
    :label: finall

    \definecolor{blue}{RGB}{18,110,213}
    \definecolor{red}{RGB}{230,0,29}
    P(\theta|D) &\propto \color{red} \left[ \frac{
     \int dz \,
    \mathcal{N} \left( \frac{ m_{B,{\rm eff}} - m_B^*(z) }{ \sqrt{ \sigma_{{\rm eff}}^2 + \sigma_{m_B}^{*2} }} \right)
    \Phi\left( \frac{m_B^*(z) - m_{B,{\rm eff}} }{ \frac{\sigma_{m_B}^{*2} +  \sigma_{{\rm eff}}^2}{\sigma_{{\rm eff}}^2} \sqrt{ \left( \frac{ \sigma_{{\rm eff}} }{ \alpha_{\rm eff} }  \right)^2 +      \frac{  \sigma_{m_B}^{*2} \sigma_{{\rm eff}}^2  }{ \sigma_{m_B}^{*2} +  \sigma_{{\rm eff}}^2 }        } }  \right)
    P(z|\theta)  }
    {\sum_{\rm passed} \frac{\mathcal{N}\left(
    \lbrace M_B, x_1, c \rbrace | \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right)}{\mathcal{N}_{\rm sim}}
    \left( \mathcal{N}_{\rm sim} dm_B\,d x_1\, d_c \right)\, dz\, dm  }\right]^N \\
    &\quad\quad\quad \color{blue} \idotsint d\vec{\eta} \,d\vec{M_B}\  \rm{Cauchy}(\sigma_{M_B}|0,2.5) \rm{Cauchy}(\sigma_{x_1}|0,2.5) \rm{Cauchy}(\sigma_{c}|0,2.5) \rm{LKJ}(\rho|4) \\
    &\quad\quad\quad \color{blue} \prod_{i=1}^N \Bigg[ \mathcal{N}\left(  \hat{\eta_i} + \delta\mathcal{Z}_b \frac{\partial\hat{\eta_i}}{\partial\mathcal{Z}}  | \eta_i, C_i \right)
    \delta\left(M_{Bi} - \left[ m_{Bi} - \mu_i + \alpha x_{1i} - \beta c_i + k(z_i) m_i \right]\right)  \\
    &\quad\quad\quad\quad\quad \color{blue}  \mathcal{N}\left( \lbrace M_{Bi}, x_{1i}, c_i \rbrace |
    \lbrace \langle M_B \rangle, \langle x_1 \rangle, \langle c \rangle \rbrace, V \right) \delta(z_i-\hat{z_i})
    \delta(m_i - \hat{m_i}) \Bigg] \\
    &\quad\quad\quad\quad\quad \color{blue}  \left[
    \int dz \,
    \mathcal{N} \left( \frac{ m_{B,{\rm eff}} - m_B^*(z) }{ \sqrt{ \sigma_{{\rm eff}}^2 + \sigma_{m_B}^{*2} }} \right)
    \Phi\left( \frac{m_B^*(z) - m_{B,{\rm eff}} }{ \frac{\sigma_{m_B}^{*2} +  \sigma_{{\rm eff}}^2}{\sigma_{{\rm eff}}^2}
    \sqrt{ \left( \frac{ \sigma_{{\rm eff}} }{ \alpha_{\rm eff} }  \right)^2 + \frac{  \sigma_{m_B}^{*2} \sigma_{{\rm eff}}^2  }{ \sigma_{m_B}^{*2} +  \sigma_{{\rm eff}}^2 }        } }  \right)
    P(z|\theta)
    \right]^{-N}\\


If you want to investigate the folders I have under dessn/models/d_simpe_stan/*, the model described
by the above equation is the ``approx_skewnorm`` folder, and gives matter density too low. The
``approx_skewnorm_rubin`` is using the same approximate correction by evaluating it at each
supernova (the exact and equal redshift approximation). ``approx`` uses the simple CDF approximation
(not the skewnorm) with a fixed CDF with. ``approx_dynamic`` then has the width of the CDF
determined properly, but adding that freedom makes :math:`\alpha` and :math:`\beta` balloon out.
``approx_mass`` is approx with mass corrections in, tested only with sncosmo and not snana to make
sure the mass didnt do anything too crazy. ``gp``, ``gp_closest`` and ``stan_mc`` are the three
methods in appendices below which Stan does not converge on. ``simple`` is a model
without approximation bias correction, and ``simple_skew`` is making the underlying colour distribution
skewed to see its effect.

The main question I want to answer is why the ``approx_skewnorm`` model is not working. To clarify
"not working", I am actually fairly happy that it seems to be working for the matter density
correction, however :math:`\beta` fits around 1.75, which is not the best. I'll put a plot of the
interim approximate surface below, ignore the truth values, they are for sncosmo and not snana. Want
:math:`\Omega_m = 0.3, \alpha=0.14, \beta=3.1, \langle M_B \rangle = -19.36, \langle x_1 \rangle = 0, \langle c \rangle = 0, \sigma_{M_B}=0.1, \sigma_{x1} = 1, \sigma_c = 0.1`.



So I ask myself, did I mess up
the math? Did I mess up the implementation with Simpson's rule? Did I mess up somewhere else in Stan?
Did I mess up with how I create the data that Stan gets? Is there some pathology I am missing?



.. figure::     ../dessn/models/d_simple_stan/approx_skewnorm/snana_dummy.png
    :align:     center
    :width:     80%

    A rough fit to five realisation of 500 SNe for the SNANA dataset. I'm pretty happy with everything bar
    :math:`\alpha` and :math:`\beta`. Showing only the approximate correction here, not the full
    Monte-Carlo correction. Want
    :math:`\Omega_m = 0.3, \alpha=0.14, \beta=3.1, \langle M_B \rangle = -19.36, \langle x_1 \rangle = 0, \langle c \rangle = 0, \sigma_{M_B}=0.1, \sigma_{x1} = 1, \sigma_c = 0.1`.



.. figure::     ../dessn/models/d_simple_stan/approx_skewnorm/zplot_approx_skewnorm_snana_dummy.png
    :align:     center
    :width:     80%

    A rough fit to five realisation of 500 SNe for the SNANA dataset, as above. However this time
    combining the chains and plotting the approximate correction in blue and the full correction in red.
    I'd need to run at least 20 realisations to be happier with the scatter, but we can see the biases
    dont look too bad in :math:`\Omega_m`.

    **It is interesting to note that :math:`\alpha` and :math:`\beta` are roughly half of what I actually want.**
    Perhaps somehow I am doing the correction twice. From fiddling with the code and running
    some more fits, I know the values for alpha and beta are highly dependent on the mean population, ie
    this line in Stan: ``cor_MB_mean = mean_MBx1c[1] - alpha*mean_MBx1c[2] + beta*mean_MBx1c[3];``. Removing
    the alpha and beta parts gives the runaway beta that I've seen in the other models, but keeping them as
    is gives alpha and beta too small.

.. code::

    \begin{table}
        \centering
        \caption{C:/Users/shint1/PycharmProjects/abc/dessn/models/d_simple_stan/approx_skewnorm}
        \label{tab:model_params}
        \begin{tabular}{cc}
            \hline
            Parameter & Corrected \\
            \hline
            $\Omega_m$ & $0.309^{+0.054}_{-0.058}$ \\
            $\alpha$ & $0.073^{+0.054}_{-0.030}$ \\
            $\beta$ & $1.72^{+0.45}_{-0.48}$ \\
            $\langle M_B \rangle$ & $-19.364^{+0.034}_{-0.028}$ \\
            $\langle x_1 \rangle$ & $-0.033^{+0.109}_{-0.076}$ \\
            $\langle c \rangle$ & $\left( -0.5^{+6.1}_{-7.3} \right) \times 10^{-3}$ \\
            $\sigma_{\rm m_B}$ & $0.190^{+0.045}_{-0.032}$ \\
            $\sigma_{x_1}$ & $0.995^{+0.083}_{-0.034}$ \\
            $\sigma_c$ & $\left( 101.6^{+5.3}_{-4.6} \right) \times 10^{-3}$ \\
            ow & $-2068.7^{+9.0}_{-16.8}$ \\
            \hline
        \end{tabular}
    \end{table}



------------------

|
|
|
|
|
|
|
|
|

Appendix 1 - MC inside Stan
---------------------------

.. warning::

    Given the concerns with the importance sampling methods, I also decided to implement
    the bias corrections within STAN itself - that is, have stan perform rough Monte-Carlo
    integration to get the bias correction in explicitly. Inserting the relevant data and structures
    into STAN such that I can perform Monte Carlo integration in a BHM framework significantly
    slows down the fits, however I believed it would at least give good results.

    .. figure::     ../dessn/models/d_simple_stan/output/plot_stan_mc_single.png
        :align:     center
        :width: 50%

        As you can see, I was wrong.

    In addition to the odd contours, we can also see in the walk itself that we have
    sampling issues, with some walkers sampling some areas of posterior space more than others.
    Stan's lack of convergence here is a big issue, indicating that the surfaces adding MC integration
    creates are intractable to stan.

    .. figure::     ../dessn/models/d_simple_stan/output/plot_stan_mc_walk.png
        :align:     center
        :width: 50%


Appendix 2 - Gaussian Processes
-------------------------------

.. warning::

    Add documentation. Conclusion is it didn't work.


Appendix 2 - Nearest Point GP
-----------------------------

.. warning::

    Add documentation. Conclusion is that is really didn't work.
"""