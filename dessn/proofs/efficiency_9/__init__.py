r""" Following from the previous example, we wish to add a second supernova type
in to the mix. From the previous example, our Ia analogues are given with the
luminosity function

.. math::
    L = L_0 \exp\left[ - \frac{(t-t_0)^2}{2s_0^2} \right].

We now introduce another functional form (our type II analogue) with luminosity

.. math::
    L = L_0 \exp\left[ - \frac{|t-t_0|}{2s_0^2} \right].

We utilise different distributions for the peak luminosity and stretch. In addition, we
also introduce an observed type. The observed type is determined to be correct 70% of the time,
such that

.. math::
    P(T_o|T) = \begin{cases}0.7 & \text{ if } T_o = T \\ 0.3 & \text{otherwise}\end{cases}

Adding in the types, the PGM now looks like such:

.. figure::     ../dessn/proofs/efficiency_9/output/pgm.png
    :width:     100%
    :align:     center

Example data that might be observed is shown below.

.. figure::     ../dessn/proofs/efficiency_9/output/data.png
    :width:     80%
    :align:     center

    Plotted here are the magnitudes for 2000 potential observations. Notice the two
    distinct luminosity populations, and how the dimmer type II analogues suffer a greater
    amount of selection bias.


"""