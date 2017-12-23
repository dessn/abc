import numpy as np
from scipy.integrate import simps
from scipy.stats import norm, skewnorm
import matplotlib.pyplot as plt

# Make true to test erf
# False to test skewnorm
if False:
    pop_mu, pop_sigma = 20, 2
    selection_mu, selection_sigma = 19, 3

    mbs = np.linspace(5, 35, 10000)
    pop_pdf = norm.pdf(mbs, pop_mu, pop_sigma)
    selection_pdf = 1 - norm.cdf(mbs, selection_mu, selection_sigma)

    plt.plot(mbs, pop_pdf)
    plt.plot(mbs, selection_pdf)

    numeric = simps(pop_pdf * selection_pdf, x=mbs)
    analytic = 1 - norm.cdf(pop_mu, selection_mu, np.sqrt(selection_sigma ** 2 + pop_sigma ** 2))
    print("Numeric  ", numeric)
    print("Analytic ", analytic)

else:
    pop_mu, pop_sigma = 20, 2

    mb_alpha = -4
    mb_mean = 21.0
    mb_width = 5

    mbs = np.linspace(5, 35, 10000)
    pop_pdf = norm.pdf(mbs, pop_mu, pop_sigma)
    selection_pdf = skewnorm.pdf(mbs, mb_alpha, mb_mean, mb_width)

    plt.plot(mbs, pop_pdf)
    plt.plot(mbs, selection_pdf)

    numeric = simps(pop_pdf * selection_pdf, x=mbs)

    mb_width2 = mb_width ** 2
    mb_alpha2 = mb_alpha ** 2
    cor_mb_width2 = pop_sigma ** 2
    mB_sgn_alpha = np.sign(mb_alpha)
    cor_sigma = np.sqrt(((cor_mb_width2 + mb_width2) / mb_width2) ** 2 * (
    (mb_width2 / mb_alpha2) + ((mb_width2 * cor_mb_width2) / (cor_mb_width2 + mb_width2))))
    cor_mb_norm_width = np.sqrt(mb_width2 + cor_mb_width2)

    analytic = np.exp(
        np.log(2) + norm.logpdf(pop_mu, mb_mean, cor_mb_norm_width) + norm.logcdf(mB_sgn_alpha * (pop_mu - mb_mean), 0,
                                                                                  cor_sigma))

    print("Numeric  ", numeric)
    print("Analytic ", analytic)
