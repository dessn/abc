r""" This module contains a simplified toy model.

We fit directly to the observed light curves using a SALT2 model.

From this, we can fit the *true* supernova parameters :math:`t_0`,
:math:`x_1` and :math:`c`. We then utilise these parameters and the global
supernova absolute magnitude :math:`M_0`, plus a realised intrinsic scatter
offset to calculate the observed distance modulus :math:`\mu_{\rm obs}`.
The observed distance modulus and absolute magnitude then gives the SALT2
parameter :math:`x_0`, which we can use in conjunction with the other SALT2
parameters to give a :math:`\chi^2` value for each supernova fit.
Graphically, the model is represented as follows.

.. figure::     ../dessn/models/a_pure_snia/output/pgm.png
    :align:     center


Using simulated data from ``SNCosmo`` to test recovery of simulation
cosmology and parameters, we find the following results.


.. figure::     ../dessn/models/a_pure_snia/output/surface.png
    :align:     center

.. figure::     ../dessn/models/a_pure_snia/output/walk.png
    :align:     center

"""