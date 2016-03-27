r""" An example used to prototype the inclusion of discrete parameters into the mix.

The scenario being modelled is thus:

A bag is filled with an infinite amount of coloured balls, either red or blue,
where the total fraction of the balls that are red is given by the variable :math:`r`. We remove only a few balls
from this infinite bag and record the colour and size of the ball.

However, the person writing down the colour is mostly colour blind, and so mistakes do happen (at a known rate).
Luckily, there is information contained in the physical size of the balls (which is measured perfectly),
as red balls are generally found to be larger in radius than the blue balls. Knowing the size distribution,
we can use this information to potentially correct for some misclassifications, and thus determine the
actual fraction :math:`r` of red balls in the bag.

Firstly, we model misidentification as:

.. math::
    P(c_i|c) = \begin{cases} 0.9,& \text{if } c_i = c\\ 0.1, & \text{otherwise} \end{cases}


We model the sizes as being Gaussian distributed based on the colour.

.. math::
    P(s_i|c) = \begin{cases}
        \frac{1}{\sqrt{2\pi}0.3} \exp\left( -\frac{(s_i - 2)^2}{2\times 0.3^2} \right)
        ,& \text{if } c = {\rm red} \\
        \frac{1}{\sqrt{2\pi}0.3} \exp\left( -\frac{(s_i - 1)^2}{2\times 0.3^2} \right),
        & \text{if } c = {\rm blue}
        \end{cases}

Following a basic binomial process, given a total fraction rate of :math:`r`, we have

.. math::
    P(c|r) = \begin{cases}
        r
        ,& \text{if } c = {\rm red} \\
        1-r,
        & \text{if } c = {\rm blue}
        \end{cases}

Putting a flat prior on the rate between 0 and 1 (:math:`P(r) = 1`), the total probability
for :math:`n` data points is

.. math::
    P(r|s_o,c_o) \propto  \prod_{i=1}^{n} \sum_c P(s_{i}|c) P(c_{i}|c) P(c|r) P(r)

With three conditional probabilities, we will have three edges in our node, one discrete parameter :math:`c`,
one underlying node :math:`r`, and two observed parameters :math:`s_o` and :math:`c_o`.
"""