r""" This module contains a simplified toy model.

Instead of performing the analysis from the observed light curves,
we instead observe the fitted parameters :math:`x_1`, :math:`m_B`
and :math:`c` and covariance :math:`\hat{C}` for each supernova.

From this, we can fit the *true* supernova parameters :math:`m_B`,
:math:`x_1` and :math:`c`. We then utilise these parameters and the global
supernova absolute magnitude :math:`M_0` to calculate the observed
distance modulus :math:`\mu_{\rm obs}`. The observed distance
modulus can then be combined with the cosmological distance modulus,
which is given as a function of the observed supernova redshift and
cosmological parameters. Graphically, the model is represented as follows.

.. figure::     ../dessn/models/c_pure_snia_simplified/output/pure_snia.png
    :align:     center


Using simulated data from ``SNCosmo`` to test recovery of simulation
cosmology and parameters, we find the following results.


.. figure::     ../dessn/models/c_pure_snia_simplified/output/surface.png
    :align:     center

.. figure::     ../dessn/models/c_pure_snia_simplified/output/walk.png
    :align:     center

"""