import numpy as np
from scipy.stats import binned_statistic


def get_data(des=True, model="G10"):
    print("Getting data for %s %s" % (("DES" if des else "LowZ"), model))
    if des:
        file = "snana_data/DES3YR_DES_BHMEFF_AM%s/passed_0.npy" % model
    else:
        file = "snana_data/DES3YR_LOWZ_BHMEFF_%s/passed_0.npy" % model
    data = np.load(file)

    result = {
        "z": data[:, 1],
        "sim_mb": data[:, 3],
        "sim_x1": data[:, 4],
        "sim_c": data[:, 5],
        "mb": data[:, 6],
        "x1": data[:, 7],
        "c": data[:, 8],
        "obs": data[:, 6:9],
        "sim": data[:, 3:6],
        "cov": data[:, 12:12+9].reshape((-1, 3, 3))
    }
    return result

if __name__ == "__main__":
    g10 = get_data(des=True, model="G10")
    c11 = get_data(des=True, model="C11")

    g10_diff = g10["obs"] - g10["sim"]
    c11_diff = c11["obs"] - c11["sim"]

    bine = 10
    g10_d_mb, bine, _ = binned_statistic(g10["z"], g10_diff[:, 0], bins=bine)
    g10_d_x1, bine, _ = binned_statistic(g10["z"], g10_diff[:, 1], bins=bine)
    g10_d_c, bine, _  = binned_statistic(g10["z"], g10_diff[:, 2], bins=bine)
    c11_d_mb, bine, _ = binned_statistic(c11["z"], c11_diff[:, 0], bins=bine)
    c11_d_x1, bine, _ = binned_statistic(c11["z"], c11_diff[:, 1], bins=bine)
    c11_d_c, bine, _  = binned_statistic(c11["z"], c11_diff[:, 2], bins=bine)
    binc = 0.5 * (bine[:-1] + bine[1:])
    print("Bias amount (100ths of a mag)")
    for i, bc in enumerate(binc):
        print("%5.3f | %6.3f %6.3f | %6.3f %6.3f | %6.3f %6.3f" % (bc, 100 * g10_d_mb[i], 100 * c11_d_mb[i], 100 * 0.1 * g10_d_x1[i], 100 * 0.1 * c11_d_x1[i], 100 * 3 * g10_d_c[i], 100 * 3 * c11_d_c[i]))

    # import matplotlib.pyplot as plt
    # plt.scatter(c11["z"], c11["c"], s=1, c='r', alpha=0.5)
    # plt.scatter(c11["z"], c11["sim_c"], s=1, c='k', alpha=0.5)
    # plt.hist(c11["c"], 50, histtype="step")
    # plt.hist(c11["sim_c"], 50, histtype="step")
    # plt.show()

    print("Diff")
    print(np.mean(g10_diff, axis=0))
    print(np.mean(c11_diff, axis=0))

    g10_cov = np.cov(g10_diff, rowvar=False)
    c11_cov = np.cov(c11_diff, rowvar=False)

    print("Calced Cov")
    print(g10_cov)
    print(c11_cov)

    g10_cov_mean = np.mean(g10["cov"], axis=0)
    c11_cov_mean = np.mean(c11["cov"], axis=0)

    print("Mean cov")
    print(g10_cov_mean)
    print(c11_cov_mean)

