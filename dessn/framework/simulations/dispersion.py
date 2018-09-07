import numpy as np


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

