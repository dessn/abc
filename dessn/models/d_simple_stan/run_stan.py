import os
import pickle
from astropy.cosmology import FlatwCDM
import numpy as np
from numpy.random import normal

def get_data():
    np.random.seed(0)
    n_sne = 500
    obs_mBx1c = []
    obs_mBx1c_cov = []
    obs_mBx1c_cor = []
    redshifts = np.linspace(0.05, 1.1, n_sne)
    MB = -19.3
    Om = 0.3
    w = -1.0
    H0 = 70.0
    intrinsic = 0.1
    cosmology = FlatwCDM(H0, Om, w0=w)
    dist_mod = cosmology.distmod(redshifts).value
    alpha = 0.1
    beta = 3
    for mu in dist_mod:
        x1 = np.random.normal(0, 1)
        c = np.random.normal(0, 0.1)
        mb = MB + mu - alpha * x1 + beta * c + normal(scale=intrinsic) + normal(scale=0.1)
        diag = np.array([0.1, 0.02, 0.02])**2
        cov = np.diag(diag)
        cor = cov / np.sqrt(np.diag(cov))[None, :] / np.sqrt(np.diag(cov))[:, None]
        obs_mBx1c_cor.append(cor)
        obs_mBx1c_cov.append(cov)
        obs_mBx1c.append([mb, x1, c])
        # print("mb is %f, x_1 is %f and c is %f mu is %f" % (mb, x1, c, mu))

    # Build a more finely sampled redshift array such that all supernova
    # redshifts fall on even indices
    n_z = 1000
    dz = redshifts.max() / n_z
    zs = sorted(redshifts.tolist())
    added_zs = [0]
    pz = 0
    for z in zs:
        est_point = int((z - pz) / dz)
        if est_point % 2 == 0:
            est_point += 1
        est_point = max(3, est_point)
        new_points = np.linspace(pz, z, est_point)[1:-1].tolist()
        added_zs += new_points
        pz = z
    n_z = n_sne + len(added_zs)
    n_simps = int((n_z + 1)/ 2)
    to_sort = [(z, -1) for z in added_zs] + [(z, i) for i, z in enumerate(redshifts)]
    to_sort.sort()
    final_redshifts = [z[0] for z in to_sort]
    sorted_vals = [(z[1], i) for i, z in enumerate(to_sort) if z[1] != -1]
    sorted_vals.sort()
    final = [int(z[1]/2 + 1) for z in sorted_vals]
    # import matplotlib.pyplot as plt
    # plt.plot(final_redshifts, np.zeros(len(final_redshifts)), 'bs', ms=10)
    # plt.plot(added_zs, np.zeros(len(added_zs)), 'g^', ms=7)
    # plt.plot(redshifts, np.zeros(len(redshifts)),  'ro')
    # plt.show()
    return {
        "n_sne": n_sne,
        "n_z": n_z,
        "n_simps": n_simps,
        "obs_mBx1c": obs_mBx1c,
        "obs_mBx1c_cov": obs_mBx1c_cov,
        "obs_mBx1c_cor": obs_mBx1c_cor,
        "zs": final_redshifts,
        "redshift_indexes": final,
        "redshifts": redshifts
        }


def init_fn():
    data = get_data()
    x1s = np.array([x[1] for x in data["obs_mBx1c"]])
    cs = np.array([x[2] for x in data["obs_mBx1c"]])
    n_sne = x1s.size
    return {
        "MB": -19.3 + normal(scale=0.1),
        "Om": 0.3 + normal(scale=0.01),
        "alpha": 0.1 + normal(scale=0.05),
        "beta": 3 + normal(scale=0.1),
        "true_c": cs + normal(scale=0.05, size=n_sne),
        "true_x1": x1s + normal(scale=0.1, size=n_sne),
        "sigma_int": np.random.uniform(0.05, 0.3)
    }


if __name__ == "__main__":
    dir_name = os.path.dirname(__file__) or "."
    output_dir = os.path.abspath(dir_name + "/output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    i = 0
    t = output_dir + "/temp%d.pkl" % i
    data = get_data()
    import pystan
    sm = pystan.StanModel(file="model.stan", model_name="Cosmology")
    fit = sm.sampling(data=data, iter=6000, warmup=1000, chains=4, init=init_fn)
    with open(t, 'wb') as output:
        dictionary = fit.extract(pars=["MB", "Om",  "alpha", "beta", "sigma_int", "PointPosteriors"])
        pickle.dump(dictionary, output)
