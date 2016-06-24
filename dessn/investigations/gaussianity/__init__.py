r"""
In most supernova cosmology analyses, the analysis is performed not using the observed
light curves, but using SALT2 summary statistics. Whilst this choice does provide
strong computational benefits, the underpinning assumption of validity for using
the summary statistics - namely that they are Gaussian - is often assumed.

In order to provide a more rigorous justification for utilising this assumption,
I investigate how well the summary statistics reflect the full posterior
surface of the SALT2 fits under a variety of conditions and supernova
parametrisation.

Introduction
------------

To begin with, we pick the optimal case for a supernova: a high signal-to-noise
signal observed regularly throughout the entire supernova lifespan for
a supernova with an exactly determined redshift. We generate
a value for supernova parameters :math:`x_0,\, x_1,\, t_0,\, c,\, z` and realise
a SALT2 light curve using ``sncosmo``. The light curve realised is shown below.


.. figure::     ../dessn/investigations/gaussianity/output/lc_simple.png
    :align:     center
    :width:     100%

    A realised light curve in the DES filters, with 30 observations in each band spanning
    a time frame of 60 days before and after the peak time.

We then fit the light curves to get summary statistics using ``sncosmo``, and compare
this to a separate MCMC fit which records the entire posterior surface. In this optimal case
we hope to see a high degree of agreement between the summary statistics and full posterior
surface.

.. figure::     ../dessn/investigations/gaussianity/output/surfaces_simple.png
    :align:     center
    :width:     100%

    The two posterior surfaces agree closely. Contours shown are for 0.5, 1, 2 and 3 sigma levels.

This simple example provides the framework for a more sophisticated analysis. In it, we vary
the supernova parameters and observation scenario (time pre-peak, and post-peak, frequency
of observations) and determine the difference between the full posterior surface and
summary statistics via a given metric, and determine in what regions of parameter space
might the gaussianity assumption not adequately hold.

Our primary parameter of interest is not any of the fit parameters, but is instead
the distance modulus :math:`\mu`, as it is biases in the distance modulus which will
bias our cosmology. Using fiducial values for :math:`\alpha` and :math:`\beta` from [1]_,
we have the distance modulus as

.. math::
    \mu = m_B^* - M + \alpha x_1 - \beta c,

where :math:`m_B^*` is calculated using ``sncosmo``. Using this relationship, and moving
the nuisance parameter :math:`M` to the other side, we can convert our multi-dimensional
posterior surface into a single dimension, and then check for any biases in the mean and
variance of the :math:`\mu+M` distribution. Doing this for the simple example above gives
us the distributions shown below, which have :math:`\Delta (\mu+M) = 0.02`
and :math:`\Delta (\sigma_{\mu+M}) = 0.004`.


.. figure::     ../dessn/investigations/gaussianity/output/mu_simple.png
    :align:     center
    :width:     50%

    The posterior surfaces from the previous figure transformed into distance modulus.




Generalisation
--------------

By finding the difference in mean and standard deviation of the :math:`\mu+M` distributions
from assuming a Gaussian approximation and examining the full posterior surface, we can
investigate the effect of various observational conditions. Parameters which may be of
interest are the peak signal to noise of the light curves, the number of observations,
and when observations for a supernova begin (ie how early do we catch the supernova
before peak) - in addition to stretch, colour and redshift.

To simplify the problem, we assume consistent observations every five days.


First, we generate supernova with stretch and colour set to zero, with observations
starting well before the peak and spanning the lifetime of the supernova. We allow
the redshift and sky flux to vary (ie redshift and signal-to-noise changes). The shift in
the mean value for :math:`\mu+M` is shown in the figure below.


.. figure::     ../dessn/investigations/gaussianity/output/bias.png
    :align:     center
    :width:     100%

    The bias in :math:`\mu+M` as a function of redshift
    and signal to noise. Each sample is shown as a point, with a third order polynomial
    fit to the surface shown as a contour. The top row represents the difference
    between the output of ``minuit`` summary statistics
    when compared to posterior surface generated using ``emcee``, and the bottom row
    shows the comparison between ``emcee`` summary statistics and the full ``emcee`` posterior.
    The left hand column shows the change in :math:`\Delta(\mu + M)`, and the right
    hand column shows the *percentage* change in standard deviation for the marginalised
    distributions of :math:`\mu+M`.

With reference to the above figure, we can see that the summary statistics provide
accurate statistics when the mean and covariance are determined from the ``emcee`` distribution,
implying low skewness of the posterior surface. However, the difference between the
``minuit`` fits and ``emcee`` fits gives rise to a difference which is not negligible.




.. [1] Betoule, M., Kessler, R., Guy, J. et al. (2014), "Improved cosmological constraints from
    a joint analysis of the SDSS-II and SNLS supernova samples",
    http://adsabs.harvard.edu/abs/2014A%26A...568A..22B

"""