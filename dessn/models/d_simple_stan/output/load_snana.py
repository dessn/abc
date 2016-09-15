# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:42:49 2016

@author: shint1
"""
import numpy as np
import pandas as pd
file = r"TEST_DATA_17677.dat"

data = pd.read_csv(file, delim_whitespace=True, comment="#", header=0)

zs = data["zCMB"].values
masses = data["HOST_LOGMASS"].values
massese = data["HOST_LOGMASS_ERR"].values

mbs = data["mB"].values
x0s = data["x0"].values
x1s = data["x1"].values
cs = data["c"].values

mbse = data["mBERR"].values
x1se = data["x1ERR"].values
cse = data["cERR"].values

cov_x1_c = data["COV_x1_c"].values
cov_x0_c = data["COV_c_x0"].values
cov_x1_x0 = data["COV_x1_x0"].values

covs = []
obs_mBx1c = []
for mb, x0, x1, c, mbe, x1e, ce, cx1c, cx0c, cx1x0 in zip(mbs, x0s, x1s, cs, mbse, x1se, cse, cov_x1_c, cov_x0_c, cov_x1_x0):
    cmbx1 = -5 * cx1x0 / (2 * x0 * np.log(10))
    cmbc = -5 * cx0c / (2 * x0 * np.log(10))
    cov = np.array([[mbe*mbe, cmbx1, cmbc], [cmbx1, x1e*x1e, cx1c], [cmbc, cx1c, ce*ce]])
    covs.append(cov)
    obs_mBx1c.append(np.array([mb, x1, c]))
    
final = {}
final["n_sne"] = mbs.size
final["obs_mBx1c"] = obs_mBx1c
final["obs_mBx1c_cov"] = covs
final["redshifts"] = zs
final["mass"] = masses

import pickle
filename = "des_sim.pickle"
pickle.dump(final, open(filename, 'wb'))