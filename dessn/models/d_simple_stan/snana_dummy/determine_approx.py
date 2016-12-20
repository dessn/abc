import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from dessn.models.d_simple_stan.get_cosmologies import get_cosmology_dictionary
from dessn.models.d_simple_stan.load import load_stan_from_folder
from dessn.models.d_simple_stan.run import get_analysis_data
from dessn.models.d_simple_stan.snana_dummy.run import get_approximate_mb_correction


def get_new_cor(chain, mean_add=0, sigma_add=0):

    mB_mean, mB_width = get_approximate_mb_correction()

    data = {
        "mB_mean": mB_mean,
        "mB_width": mB_width,
        "snana_dummy": True,
        "sim": False
    }
    d = get_analysis_data(**data)
    redshifts = d["redshifts"]
    cosmologies = get_cosmology_dictionary()

    weight = []
    for i in range(chain["mean_MB"].size):
        om = np.round(chain["Om"][i], decimals=3)
        key = "%0.3f" % om
        mus = cosmologies[key](redshifts)

        mb = chain["mean_MB"][i] + mus - chain["alpha"][i] * chain["mean_x1"][i] + chain["beta"][i] * chain["mean_c"][i]

        # cc = 1 - norm.cdf(mb, mB_mean, mB_width) + 0.001
        cc = 1 - norm.cdf(mb, mB_mean + mean_add, mB_width + sigma_add)
        w = np.sum(np.log(cc) - chain["mean_c"][i])
        weight.append(w)
    return np.array(weight)

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    std = dir_name + "/stan_output"
    res = load_stan_from_folder(std, replace=False, merge=True, cut=False)
    chain, posterior, t, p, f, l, w, ow = res

    logw = np.log(w)
    full_log_correction = logw + ow

    if False:
        mean = np.linspace(-2, 2, 17)
        sigma = np.linspace(-2, 2, 17)
        print(mean)
        print(sigma)
        ms = []
        ss = []
        ws = []
        for m in mean:
            for s in sigma:
                w = get_new_cor(chain, mean_add=m, sigma_add=s)
                w = full_log_correction - w
                w -= w.mean()
                ms.append(m)
                ss.append(s)
                ws.append(np.std(w))
        ws = np.array(ws)
        ms = np.array(ms)
        ss = np.array(ss)
        iis = np.argsort(ws)

        for i in iis:
            print(ms[i], ss[i], ws[i])

    new_cor = get_new_cor(chain, mean_add=1.5, sigma_add=-1.0)
    diff = full_log_correction - new_cor
    diff2 = full_log_correction - ow
    diff3 = full_log_correction

    diff -= diff.mean()
    diff2 -= diff2.mean()
    diff3 -= diff3.mean()

    plt.hist(diff, 100, histtype="step", label="new")
    plt.hist(diff2, 100, histtype="step", label="Original")
    plt.hist(diff3, 100, histtype="step", label="No approx")
    plt.legend()
    plt.show()