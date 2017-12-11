import os
import inspect
import numpy as np
from dessn.blind.blind import blind_om, blind_w


def read(chainsfile,blind=True):
    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    chainsData = np.load(this_dir + "/" + chainsfile)
    
    samples = chainsData.samples
    indices = chainsData.index
    
    omegam = samples[indices['omegam']]
    w = samples[indices['w']]
    if blind:
        omegam = blind_om(omegam)
        w = blind_w(w)
    
    return omegam, w
    
#chainsFile = '/scratch/midway/rkessler/djbrout/cosmomc/chains2/DB17_ALL_omw_alone.py_mcsamples'
#omegam,w = read(chainsFile,blind=False)    
#print omegam
