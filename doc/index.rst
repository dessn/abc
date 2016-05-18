************************
DESSN Cosmology Analysis
************************


This document should detail the code structure, examples, and all other goodies.


Examples
--------

Learning from examples is my preferred method.

For the primary motivation example, explaining how the models work and tie together, and why
certain design choices have been chosen, see the :py:mod:`dessn.examples.simple` package.



There are two example models which use discrete parameters (as they can be used in two
different ways), and these are detailed in :py:mod:`dessn.examples.discrete` and
:py:mod:`dessn.examples.discrete_observed`.


Core
----

To learn how the underlying models function, and specific details on the sorts of parameters
and edges allowed, please see the documentation located at :py:mod:`dessn.framework`.


Utilities
---------

For examples and usage instructions on how to use the plotting library included in this project,
see :py:mod:`dessn.chain`.


Implementations
---------------

Finally, for the toy model implementations, see :py:mod:`dessn.models`.

Specially, for the
working toy model implementation going from 'observed' SALT2 parameters, see
:py:mod:`dessn.models.c_pure_snia_simplified`. For an implementation using the observed
light curves instead of SALT2 parameters, see :py:mod:`dessn.models.a_pure_snia`.


General
-------
The general project structure is as follows:

.. toctree::
   :maxdepth: 4

   dessn
