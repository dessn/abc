import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm, linregress


def get_simulation_data():
    pickle_file = "../output/supernovae2.npy"
    supernovae = np.load(pickle_file)

    return {
        "n_sim": supernovae.shape[0],
        "sim_MB": supernovae[:, 0],
        "sim_mB": supernovae[:, 1],
        "sim_x1": supernovae[:, 2],
        "sim_c": supernovae[:, 3],
        "sim_passed": supernovae[:, 6],
        "sim_log_prob": supernovae[:, 7],
        "sim_redshift": supernovae[:, 5],
        "sim_mass": supernovae[:, 4]
    }


def get_approximate_mb_correction(plot=False):
    d = get_simulation_data()
    mask = d["sim_passed"] == 1
    z = d["sim_redshift"]
    m = d["sim_mass"]
    mB = d["sim_mB"]
    MB = d["sim_MB"]
    x1 = d["sim_x1"]
    c = d["sim_c"]

    hist_all, bins = np.histogram(mB, bins=200)
    hist_passed, _ = np.histogram(mB[mask], bins=bins)
    binc = 0.5 * (bins[:-1] + bins[1:])
    ratio = 1.0 * hist_passed / hist_all
    inter = interp1d(ratio, binc)
    mean = inter(0.5)
    width = 0.5 * (inter(0.16) - inter(0.84))

    # hist_all2, bins2 = np.histogram(c, bins=200)
    # hist_passed2, _ = np.histogram(c[mask], bins=bins2)
    # binc2 = 0.5 * (bins2[:-1] + bins2[1:])
    # cmask = (binc2 > -0.1) & (binc2 < 0.3)
    # ratio2 = 1.0 * hist_passed2 / hist_all2
    # ratio2 = ratio2[cmask]
    # binc2 = binc2[cmask]
    # slope, intercept, r_value, p_value, std_err = linregress(binc2, ratio2)
    # zero = (-intercept / slope)
    # assert zero > 0.6
    # print(slope, intercept, zero)

    adj = mB
    hist_all, binsx, binsy = np.histogram2d(adj, c, bins=50)
    hist_passed, _, _ = np.histogram2d(adj[mask], c[mask], bins=[binsx, binsy])
    bincx = 0.5 * (binsx[:-1] + binsx[1:])
    bincy = 0.5 * (binsy[:-1] + binsy[1:])
    cratio = 1.0 * hist_passed / hist_all
    cratio = np.ma.masked_where(hist_all == 0, cratio)
    hist_all = np.ma.masked_where(hist_all == 0, hist_all)
    hist_passed = np.ma.masked_where(hist_all == 0, hist_passed)

    if plot:
        fig, ax = plt.subplots(nrows=2, ncols=2)
        h = ax[0, 0].pcolormesh(bincx, bincy, cratio.T, cmap='viridis')
        ax[1, 0].pcolormesh(bincx, bincy, hist_passed.T)
        ax[1, 1].pcolormesh(bincx, bincy, hist_all.T)
        fig.colorbar(h)
        plt.show()

        cdf = 1 - norm.cdf(binc, mean, width)
        fig, ax = plt.subplots(5)
        ax[0].hist(mB, 50, histtype='step')
        ax[0].hist(mB[mask], 50, histtype='step')
        ax[1].hist(x1, 50, histtype='step')
        ax[1].hist(x1[mask], 50, histtype='step')
        ax[2].hist(c, 50, histtype='step')
        ax[2].hist(c[mask], 50, histtype='step')
        ax[2].axvline(0.08, c='g')
        ax[2].axvline(0.1, c='b')
        ax[3].plot(binc, ratio)
        ax[3].plot(binc, cdf)
        # ax[4].plot(binc2, ratio2)
        # xs = np.linspace(binc2.min(), binc2.max(), 10)
        # ys = intercept + slope * xs
        # ax[4].plot(xs, ys)
        plt.show()
    return mean, width #, slope, intercept

if __name__ == "__main__":
    print(get_approximate_mb_correction(plot=True))
