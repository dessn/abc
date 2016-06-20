"""
In most supernova cosmology analyses, the analysis is performed not using the observed
light curves, but using SALT2 summary statistics. Whilst this choice does provide
strong computational benefits, the underpinning assumption of validity for using
the summary statistics - namely that they are Gaussian - is often assumed.

In order to provide a more rigorous justification for utilising this assumption,
I investigate how well the summary statistics reflect the full posterior
surface of the SALT2 fits under a variety of conditions and supernova
parametrisation.

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

"""