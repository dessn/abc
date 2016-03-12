import numpy as np


def plus(loga, logb):
    """ Returns :math:`log(a + b)` when given :math:`log(a)` and :math:`log(b)`.
    """
    if loga > logb:
        return loga + np.log(1 + np.exp(logb - loga))
    else:
        return logb + np.log(1 + np.exp(loga - logb))
