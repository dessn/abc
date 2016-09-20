import numpy as np
from astropy.cosmology import FlatwCDM

# These are the rates from the SNANA input files.
# DNDZ:  POWERLAW2  2.60E-5  1.5  0.0 1.0  # R0(1+z)^Beta Zmin-Zmax
# DNDZ:  POWERLAW2  7.35E-5  0.0  1.0 2.0

def rate(z):
    zlo = z < 1
    dndz = zlo * 2.6e-5 * (1 + z)**1.5 + (1 - zlo) * 7.35e-5 * (1 + z)
    return dndz