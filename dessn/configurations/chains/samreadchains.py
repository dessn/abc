import os
import inspect
import pickle
from dessn.blind.blind import blind_om, blind_w


def read(chain_file, blind=True):

    this_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

    with open(this_dir + "/" + chain_file, 'rb') as f:
        chain_data = pickle.load(f, encoding='latin1')

    samples = chain_data.samples
    indices = chain_data.index
    omegam = samples[:, indices['omegam']]
    w = samples[:, indices['w']]
    if blind:
        omegam = blind_om(omegam)
        w = blind_w(w)

    norm_weight = 100000.0 * chain_data.weights / chain_data.weights.sum()
    return omegam, w, norm_weight
