************************
DESSN Cosmology Analysis
************************


This document should detail the code structure, examples, and all other goodies.

.. _examples:

Examples
--------

Learning from examples is my preferred method.

For the primary motivation example, explaining how the models work and tie together, and why
certain design choices have been chosen, see the :py:mod:`dessn.examples.simple` package.
There are two example models which use discrete parameters (as they can be used in two
different ways), and these are detailed in :py:mod:`dessn.examples.discrete` and
:py:mod:`dessn.examples.discrete_observed`.

.. table::
    :class: borderless

    +----------------------------------------------------------------+------------------------------------------------------------------+-----------------------------------------------------------------------+
    |..  image:: ../dessn/examples/simple/output/surfaces.png        |..  image:: ../dessn/examples/discrete/output/surfaces.png        |..  image:: ../dessn/examples/discrete_observed/output/surfaces.png    |
    |    :width: 95%                                                 |    :width: 95%                                                   |    :width: 95%                                                        |
    |    :align: center                                              |    :align: center                                                |    :align: center                                                     |
    |    :target: example_simple.html                                |    :target: example_discrete.html                                |    :target: example_discrete_observed.html                            |
    |                                                                |                                                                  |                                                                       |
    |:ref:`example_simple`                                           |:ref:`example_discrete`                                           |:ref:`example_discrete_observed`                                       |
    +----------------------------------------------------------------+------------------------------------------------------------------+-----------------------------------------------------------------------+


------------

Chain Consumer
--------------

Also written for this project is a robust tool that consumes chains and produces plots.
This tool is detailed at :py:mod:`dessn.chain`.


Core
----

To learn how the underlying models function, and specific details on the sorts of parameters
and edges allowed, please see the documentation located at :py:mod:`dessn.framework`.


Utilities
---------

Stuff


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
   :maxdepth: 2

   examples
   chain
