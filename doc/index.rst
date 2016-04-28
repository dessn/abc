************************
DESSN Cosmology Analysis
************************


This document should detail the code structure, examples, and all other goodies.


Examples
--------

Learning from examples is my preferred method.

For the primary motivation example, explaining how the models work and tie together, and why
certain design choices have been chosen - primarily marginalisation over numerical integration,
see the :py:mod:`dessn.examples.simple` package.



For a different example, which shows how to use discrete parameters, see the :py:mod:`dessn.examples.discrete` package.


Core
----

To learn how the underlying models function, and specific details on the sorts of parameters and edges allowed,
please see the documentation located at :py:mod:`dessn.model`.


Utilities
---------

For examples and usage instructions on how to use the plotting library included in this project,
see :py:mod:`dessn.chain`.


Implementations
---------------

Finally, for the toy model implementation, see :py:mod:`dessn.toy`


General
-------
The general project structure is as follows:

.. toctree::
   :maxdepth: 4

   dessn
