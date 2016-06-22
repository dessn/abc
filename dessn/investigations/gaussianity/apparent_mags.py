import sncosmo
import numpy as np
import os
from scipy.interpolate import RectBivariateSpline


def generate_and_return():
    filename = os.path.dirname(__file__) + "/output/apparent.npy"
    x1s = np.linspace(-5, 5, 200)
    cs = np.linspace(-1, 1, 200)

    if not os.path.exists(filename):
        model = sncosmo.Model(source='salt2')
        apparents = np.zeros((x1s.size, cs.size))
        for i, x1 in enumerate(x1s):
            for j, c in enumerate(cs):
                apparents[i, j] = get_apparent(model, x1, c)
                print(i, apparents[i, j])
        np.save(filename, apparents)
    else:
        apparents = np.load(filename)
    return RectBivariateSpline(x1s, cs, apparents)


def get_apparent(model, x1, c):
    params = {"t0": 1000, "z": 0.1, "x0": 1e-5, "x1": x1, "c": c}
    model.set(**params)
    return model.source.peakmag("bessellb", "ab")

if __name__ == "__main__":
    generate_and_return()

