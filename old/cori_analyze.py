import pickle
import corner
import matplotlib.pyplot as plt
import numpy

class Inputs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

inputs = Inputs(Om0 = 0.28, w0=-1., rate_II_r=2., logL_snIa=numpy.log(1.), sigma_snIa=0.01, logL_snII = numpy.log(0.5), sigma_snII=0.4, Z=0.)



pkl_file = open('data.pkl', 'rb')
data = pickle.load(pkl_file)
samples = data[:, 800:, :].reshape((-1, data.shape[2]))
pkl_file.close()
fig = corner.corner(samples[:,:8], labels=["$\Omega_M$", "$w_0$", "$r_{snII}$", "$\ln{L_{snIa}}$","$\sigma_{snIa}$", "$\log{L_{snII}}$", "$\sigma_{snII}$", "$Z$"],  truths=[inputs.Om0, inputs.w0, inputs.rate_II_r, inputs.logL_snIa, inputs.sigma_snIa, inputs.logL_snII, inputs.sigma_snII, inputs.Z])
fig.savefig("triangle.png")

