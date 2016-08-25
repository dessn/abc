r"""

A fully rigorous analysis would be expected to integrate zero point
uncertainty and covariance by having the conversion between counts
and flux. However, this is numerically infeasible, as there are so
many observational effects that must be included to get the actual
flux and flux error. Because of this, zero point uncertainty is
generally propagated into analysis by determining the numerical
derivative of the parameters of interest (generally apparent magnitude,
stretch and colour of the supernovae) with respect to the zero points
by simulations. In doing this, there is an assumption made about the
Gaussianity of the uncertainties, namely that the the posterior
in a full analysis remains Gaussian.

We inspect this assumption for the case of a single supernova, where
we fit not only the SALT2 parameters but the zero points simultaneously,
where the zero points are marginalised. This gives us a full posterior
surface which takes into account zero point uncertainties rigorously.

We then determine the SALT2 parameter surface using the traditional
method, and compare the resulting distributions in the hope that
differences are negligible.
"""