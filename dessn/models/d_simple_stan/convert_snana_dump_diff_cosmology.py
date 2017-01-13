# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:42:49 2016

@author: shint1
"""
import numpy as np
import pandas as pd
import os
import inspect

from numpy.lib.recfunctions import drop_fields, append_fields

if __name__ == "__main__":
    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)
    data_file = os.path.abspath(dir_name + "/data/SHINTON_DIFF_COSMOLOGY.DUMP")
    output_file = os.path.abspath(dir_name + "/data/snana_diff_cosmology.npy")

    print("Reading %s" % data_file)
    dataframe = pd.read_csv(data_file, sep='\s+', skiprows=1, comment="#")

    supernovae = dataframe.to_records()

    cutmask = (supernovae["CUTMASK"]) > 1022
    supernovae = drop_fields(supernovae, "CUTMASK")
    supernovae = drop_fields(supernovae, "S2x0")
    supernovae = drop_fields(supernovae, "S2alpha")
    supernovae = drop_fields(supernovae, "S2beta")
    supernovae = drop_fields(supernovae, "VARNAMES:")
    supernovae = drop_fields(supernovae, "index")
    print(supernovae.dtype)
    supernovae = supernovae.view((supernovae.dtype[0], len(supernovae.dtype.names)))
    supernovae = supernovae.astype(np.float32)[cutmask, :]
    print(supernovae.shape)
    np.save(output_file, supernovae)
    print("Conversion saved to %s" % output_file)
