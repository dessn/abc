""" I have placed the class based example for implementing the simplified model
into its own module, so that the documentation generating for the ``simple`` module
does not get cluttered with all the small classes this module will have.

The primary class to look at in code is the :class:`.ExampleModel` class.

I should finally note that in order to demonstrate parameter transformations, I
have modified the model used in the previous two examples (:class:`.ExampleIntegral`
and :class:`.ExampleLatent`) to also include a luminosity transformation, where I
simply halve the luminosity before converting it to flux. Physically, this could
represent a perfect 50% mirror absorption on the primary telescope mirror.
"""