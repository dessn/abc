************************
DESSN Cosmology Analysis
************************


This document should detail the code structure, examples, and all other goodies.


Model Proofs
============

.. include:: proofs.rst


Investigations
==============

.. include:: investigations.rst

Implementations
===============

.. include:: implementations.rst



General
=======

The general project structure is as follows:

.. toctree::
    :maxdepth: 2

    examples_sep
    chain
    core
    proofs_sep
    investigations_sep
    implementations_sep


--------------

I also implemented a general BHM framework which I think is pretty nifty,
but needed to move to STAN so its being unused at the moment.


.. _examples:

Examples
========

.. include:: examples.rst


Core Functionality
==================

To learn how the underlying models function, and specific details on the sorts of parameters
and edges allowed, please see the documentation located at :ref:`core`.
