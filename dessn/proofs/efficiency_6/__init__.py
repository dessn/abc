r""" In this example we add in multiple bands and flux calibration.

As before, we observe "supernova" from an underlying distribution, and the
absolute luminosity to apparent luminosity via some (different) distance
relation, which we shall cringe and denote redshift. The underlying zero points :math:`Z_i` are
informed by a strong calibration prior on zero points :math:`Z_o` and covariance :math:`C`.
Using the zero points and apparent magnitude (ie the flux), we can predict counts
for each band (which I will treat as an observed quantity). Using then the predicted
photon counts, which also gives us the error via the Poisson process, we can calculate the
likelihood of our observations. We assume the same apparent magnitude in all bands,
for clarification.

As I now use subscripts to denote different bands, with :math:`i` representing
all different bands (like Einstein notation), and so observed quantities will
be represented by a hat and vector quantities shall be represented by bold font.

.. math::
    \mu &\sim \mathcal{U}(500, 1500) \\
    \sigma &\sim \mathcal{U}(50, 150) \\
    Z_i &\sim \mathcal{N}(Z_o, C) \\
    L &\sim \mathcal{N}(\mu, \sigma) \\
    z &\sim \mathcal{U}(0.5, 1.5) \\
    f &= \frac{L}{z^2} \\
    c_i &= 10^{Z_i / 2.5} f \\
    \mathbf{\hat{c}_i} &\sim \mathcal{N}(c_i, \sqrt{c_i})

We create a data set by drawing from these distributions and introducing our
data selection cuts. For 400 events, this gives us the following data distribution
in redshift and luminosity.

.. figure::     ../dessn/proofs/efficiency_6/output/data.png
    :width:     80%
    :align:     center



Denoting the selection effects - a signal to noise cut in all bands - this
time as :math:`S_2` we have:

.. math::
    \mathcal{L} &= P(\mathbf{\hat{c}_i}, \hat{z}|S_2,\mu,\sigma, Z_i) \\
    &= \frac{P(\mathbf{\hat{c}_i}, \hat{z}, S_2|\mu,\sigma, Z_i)}{P(S_2|\mu,\sigma, Z_i)} \\
    &= \frac{\int dL P(\mathbf{\hat{c}_i}, \hat{z}, S_2, L|\mu,\sigma, Z_i)}{P(S_2|\mu,\sigma, Z_i)} \\
    &= \frac{\int dL \ P(S_2 | \mathbf{\hat{c}_i}) P(\mathbf{\hat{c}_i}| L \hat{z}, Z_i)
    P(\hat{z}) P(L|\mu,\sigma)}{P(S_2|\mu,\sigma, Z_i)}

Here we also assume flat priors on redshift, and as our
data has already gone through the selection cuts, :math:`P(S_2|\mathbf{\hat{c}_i}) = 1`.

.. math::
    \mathcal{L} &= \frac{\int dL \  P(\mathbf{\hat{c}_i}| L, Z_i, \hat{z})
    P(L|\mu,\sigma)}{P(S_2|\mu,\sigma, Z_i)} \\
    &= \frac{\int dL \  P(\mathbf{\hat{c}_i}| L, \hat{z}, Z_i)
    P(L|\mu,\sigma)}{P(S_2|\mu,\sigma, Z_i)} \\
    &= \frac{\int dL \ \mathcal{N}\left(\mathbf{\hat{c}_i}; 10^{Z_i/2.5} \frac{L}{\hat{z}^2}, \sqrt{10^{Z_i/2.5}\frac{L}{\hat{z}^2}}\right)
    \mathcal{N}\left(L ; \mu, \sigma\right)}   {P(S_2|\mu,\sigma,Z_i)} \\

Finally, as flux is easier to estimate than luminosity for a starting position,
we transform our integral over luminosity into an integral over flux.
Which gives us :math:`dL = \hat{z}^2 df`, leading to:

.. math::
    \mathcal{L} &= \frac{\int df\ \hat{z}^2 \ \mathcal{N}\left(\mathbf{\hat{c}_i}; 10^{Z_i/2.5} f, \sqrt{10^{Z_i/2.5}f}\right)
    \mathcal{N}\left(\frac{f}{\hat{z}^2} ; \mu, \sigma\right)}   {P(S_2|\mu,\sigma,Z_i)} \\

We now need to go through the fun process of determining the model efficiency in the
denominator. Due to the presence of multiple bands, this is more complicated than
the previous example. We consider the case where we have :math:`N` observations
in each band, and we only have two bands, denoted with a subscript :math:`1` and :math:`2`
respectively.

.. math::
    P(S_2 | \mu, \sigma, Z_i) &= \int dL \int dz \int d\mathbf{c_i} \ P(S_2, \mathbf{c_i}, L, z | \mu, \sigma, Z_i) \\
    &= \int dz \int dL \int d\mathbf{c_1} \int d\mathbf{c_2} \ P(S_2, \mathbf{c_1}, \mathbf{c_2}, L, z | \mu, \sigma, Z_1, Z_2) \\
    &= \int dz \int dL \int d\mathbf{c_1} \int d\mathbf{c_2} \ P(S_2 | \mathbf{c_1}, \mathbf{c_2}) P( \mathbf{c_1}| L, z, Z_1) P( \mathbf{c_2}| L, z, Z_2) P(L| \mu, \sigma) P(z) \\

Again enforcing the flat prior on redshift, and converting from luminosity to flux:

.. math::
    P(S_2 | \mu, \sigma, Z_i) &= \int dz\, z^2 \int df P(fz^{-2}| \mu, \sigma) \int d\mathbf{c_1} \int d\mathbf{c_2} \ P(S_2 | \mathbf{c_1}, \mathbf{c_2}) P( \mathbf{c_1}| f, Z_1) P( \mathbf{c_2}| f, Z_2) \\

Translating to english, :math:`P(S_2 | \mathbf{c_1}, \mathbf{c_2})` is the probability that
either of the bands has at least 2 points above a specified signal to noise cut, such that
the counts is greater than some number :math:`\alpha^2`. This can be inverted to turn
the *or* condition into an *and*, which is simpler to calculate.

.. math::
    P(S_2 |  \mathbf{c_1}, \mathbf{c_2}) = 1 - P(\bar{S_2} |  \mathbf{c_1}, \mathbf{c_2})

We can further note that the probability the data does not pass selection cuts is separable
into the probability band 1 fails *and* band 2 fails.

.. math::
    P(S_2 | \mathbf{c_1}, \mathbf{c_2}) = 1 - P(\bar{S}_{2,1} |  \mathbf{c_1}) P(\bar{S}_{2,2} | \mathbf{c_2})

We can separate this again: the probability band 1 fails selection is sum of probability that
zero points meet the signal-to-noise criterion or only one points meets the criterion.

.. math::
    P(\bar{S}_{2,1} | \mathbf{c_1}) = P(S_{0,1} | \mathbf{c_1}) + P(S_{1,1}|\mathbf{c_1})

Let's continue breaking this down. The probability that we have 0 points above a
signal-to-noise cut, due to the independence of successive points, can be written as a product.

.. math::
    P(S_{0,1}|\mathbf{c_1}) = \prod_i P(c_{1,i} < \alpha^2)

We can also see that the probability of only one point being above a signal-to-noise cut is the
same as can be found in the previous example.

.. math::
    P(S_{1,1} | \mathbf{c_1}) = \sum_i P(c_{1,i} > \alpha^2) \prod_{j\neq i} P(c_{1,j} < \alpha^2)

Putting these together:

.. math::
    P(S_2 | \mathbf{c_1}, \mathbf{c_2}) &= 1 - \left[\prod_i P(c_{1,i} < \alpha^2) + \sum_j P(c_{1,j} > \alpha^2) \prod_{k\neq j} P(c_{1,k} < \alpha^2) \right]\\
    &\quad\quad \left[ \prod_l P(c_{2,l} < \alpha^2) + \sum_m P(c_{2,m} > \alpha^2) \prod_{n\neq m} P(c_{2,n} < \alpha^2)   \right] \\
    &= 1 -  \left[ \prod_i P(c_{1,i} < \alpha^2) \right] \left[\prod_l P(c_{2,l} < \alpha^2)  \right]-\\
    &\quad\quad \left[ \prod_i P(c_{1,i} < \alpha^2) \right] \left[ \sum_m P(c_{2,m} > \alpha^2) \prod_{n\neq m} P(c_{2,n} < \alpha^2)   \right] \\
    &\quad\quad \left[ \sum_j P(c_{1,j} > \alpha^2) \prod_{k\neq j} P(c_{1,k} < \alpha^2) \right] \left[ \prod_l P(c_{2,l} < \alpha^2) \right]  \\
    &\quad\quad \left[ \sum_j P(c_{1,j} > \alpha^2) \prod_{k\neq j} P(c_{1,k} < \alpha^2) \right] \left[ \sum_m P(c_{2,m} > \alpha^2) \prod_{n\neq m} P(c_{2,n} < \alpha^2)   \right]

What an ugly term to write out! Lets denote these four terms with :math:`t`, such that the
above equation can be written
as :math:`P(S_2 | \mathbf{c_1}, \mathbf{c_2}) = 1 - (t_1 + t_2 + t_3 + t_4)`.

To put this back inside our efficiency term, we have

.. math::
    P(S_2 | \mu, \sigma, Z_i) &= 1 - \int dz\, z^2 \int df P(fz^{-2}| \mu, \sigma) \int d\mathbf{c_0} \int d\mathbf{c_1} \ (t_1 + t_2 + t_3 + t_4) P( \mathbf{c_1}| f, Z_1) P( \mathbf{c_2}| f, Z_2) \\

Now let's break this down by :math:`t`.

.. math::
    x_1 &\equiv \int d\mathbf{c_0} \int d\mathbf{c_1} t_1 P( \mathbf{c_1}| f, Z_1) P( \mathbf{c_2}| f, Z_2) \\
    &=  \int d\mathbf{c_1} \int d\mathbf{c_2} \left[ \prod_i P(c_{1,i} < \alpha^2) \right] \left[\prod_l P(c_{2,l} < \alpha^2)  \right] P( \mathbf{c_1}| f, Z_1) P( \mathbf{c_2}| f, Z_2) \\
    &=  \int d\mathbf{c_1} \left[ \prod_i P(c_{1,i} < \alpha^2) \right] P( \mathbf{c_1}| f, Z_1) \int d\mathbf{c_2}  \left[\prod_l P(c_{2,l} < \alpha^2)  \right]  P( \mathbf{c_2}| f, Z_2) \\
    &= \left[ g_{-} (f, Z_1, \alpha)  g_{-} (f, Z_2, \alpha) \right]^N,

where :math:`g_{-}` is defined by

.. math::
    g_{-}(f, Z, \alpha) &= \begin{cases}
    \frac{1}{2} + \frac{1}{2}{\rm erf} \left[ \frac{\alpha^2 - 10^{Z/2.5}f}{\sqrt{2f\cdot 10^{Z/2.5}}} \right] &
    \text{ if } \alpha^2 - 10^{Z/2.5}f > 0 \\
    \frac{1}{2} - \frac{1}{2}{\rm erf} \left[ \frac{10^{Z/2.5}f - \alpha^2}{\sqrt{2f\cdot 10^{Z/2.5}}} \right] &
    \text{ if } \alpha^2 - 10^{Z/2.5}f < 0 \\
    \end{cases} \\

For a full derivation of the above, consult previous examples.

We can also sub in the results for :math:`t_2,\ t_3,\ t_4`, as will get

.. math::
    t_1 & \rightarrow x_1 \equiv \left[ g_{-} (f, Z_1, \alpha)  g_{-} (f, Z_2, \alpha) \right]^N \\
    t_2 & \rightarrow x_2 \equiv N g_{+}(f, Z_2, \alpha) \left[g_{-}(f, Z_1, \alpha)\right]^N \left[ g_{-}(f, Z_2, \alpha) \right]^{N-1} \\
    t_3 & \rightarrow x_3 \equiv N g_{+}(f, Z_1, \alpha) \left[g_{-}(f, Z_2, \alpha)\right]^N \left[ g_{-}(f, Z_1, \alpha) \right]^{N-1} \\
    t_4 & \rightarrow x_4 \equiv N^2 g_{+}(f, Z_1, \alpha) g_{+}(f, Z_2, \alpha) \left[g_{-}(f, Z_1, \alpha) g_{-}(f, Z_2, \alpha) \right]^{N-1}

These functions are calculable, and from them we can determine the efficiency.

.. math::
    P(S_2 | \mu, \sigma, Z_i) = 1 - \int dz\, z^2 \int df \, \mathcal{N}(fz^{-2}; \mu, \sigma) \left[ x_1 + x_2 + x_3 + x_4 \right]


The model PGM is constructed as follows:

.. figure::     ../dessn/proofs/efficiency_6/output/pgm.png
    :width:     100%
    :align:     center


The weights (efficiency, denominator, or mathematically :math:`P(S_2|\mu,\sigma,Z_i)`),
sliced down a single value of :math:`Z_0` and :math:`Z_1`, appear as follows:

.. figure::     ../dessn/proofs/efficiency_6/output/weights.png
    :width:     80%
    :align:     center


Here are plotted the likelihood surfaces. Green represents all the data, including those
that didn'nt make the cut, using a model without a bias correction. Red is the data
after the cut (less data points), using a model without the bias correction. Blue is with the
biased data, using a model implementing the bias correction.

.. figure::     ../dessn/proofs/efficiency_6/output/surfaces.png
    :align:     center
    :width:     100%
"""