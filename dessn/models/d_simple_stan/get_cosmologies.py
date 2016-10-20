import numpy as np
from astropy.cosmology import FlatwCDM
from scipy.interpolate import interp1d
import inspect
import os
import pickle
from joblib import Parallel, delayed


def get_cosmology_dictionary():
    this_file = inspect.stack()[0][1]
    cosmology_pickle = os.path.dirname(this_file) + os.sep + "output/cosmologies.pkl"
    if os.path.exists(cosmology_pickle):
        with open(cosmology_pickle, 'rb') as f:
            cosmology_dictionary = pickle.load(f)
    else:
        cosmology_dictionary = generate_cosmology_dictionary()
        with open(cosmology_pickle, 'wb') as output:
            pickle.dump(cosmology_dictionary, output)
    return cosmology_dictionary


def generate_cosmology_dictionary():
    oms = np.arange(0, 1, 0.001)
    n = oms.size
    print("Generating %d cosmologies" % n)
    cosmologies = Parallel(n_jobs=4, verbose=100, batch_size=100)(delayed(get_cosmology)(om) for om in oms)
    keys = ["%0.3f" % om for om in oms]
    cosmology_dictionary = {key: cosmology for key, cosmology in zip(keys, cosmologies)}
    return cosmology_dictionary


def get_cosmology(om):
    zs = np.linspace(0, 1.2, 10000)
    mus = FlatwCDM(70.0, om).distmod(zs).value
    return interp1d(zs, mus, assume_sorted=True)