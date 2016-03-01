import logging
import numpy as np


class ObservationFactory(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        self.check_kwargs()

    def check_kwargs(self):
        reqs = ['rate_II_r', 'logL_snIa', 'logL_snII', 'sigma_snIa', 'sigma_snII', 'cosmology']
        keys = self.kwargs.keys()
        fail = []
        for req in reqs:
            if req not in keys:
                fail.append(req)
        if len(fail) > 0:
            message = "Required keys %s not found" % fail
            self.logger.error(message, exc_info=True)
            raise ValueError("Required keys %s not found" % fail)

    def get_observations(self, num):
        """ Still needs massive refactoring """
        specz = np.random.uniform(low=0.1, high=0.8, size=num)
        zprob = np.zeros(num) + 1.0

        snIa_rate = 1.0 / (1 + self.kwargs['rate_II_r'])

        spectype = (np.random.uniform(low=0, high=1, size=num) > snIa_rate) * 1
        # 0 represents snIa, 1 represents snII

        snIa_luminosity = np.exp(self.kwargs['logL_snIa'] * np.power(10, np.random.normal(0, self.kwargs['sigma_snIa']/2.5), size=num))
        snII_luminosity = np.exp(self.kwargs['logL_snII'] * np.power(10, np.random.normal(0, self.kwargs['sigma_snII']/2.5), size=num))

        luminosity = (1 - spectype) * snIa_luminosity + spectype * snII_luminosity
        luminosity_distance = self.kwargs['cosmology'].luminosity_distance(specz).value

        npts = 2
        cov = np.zeros((2,2))
        cov[0,0] = 1e-20
        cov[1,1] = 1e-20
        cov[0,1] = 0     #for now uncorrelated as algorithm handles that
        cov[1,0] = 0     #for now uncorrelated as algorithm handles that
        invcov = np.linalg.inv(cov)

        counts = []
        counts_invcov = []
        for i in xrange(num):
            ans = np.random.multivariate_normal(np.zeros(npts)+ luminosity[i] / (4 * np.pi * luminosity_distance[i] * luminosity_distance[i] * 10**(self.kwargs['Z']/2.5)), cov)
            counts.append(ans)
            counts_invcov.append(invcov)

        nthreshold = 1
        found = []

        fluxthreshold = 0.4e-8

        for i in xrange(num):
            nabove = (counts[i] >= fluxthreshold).sum()
            found.append(nabove >= nthreshold)
        found = np.array(found)

        num = found.sum()
        specz = [np.array([dum]) for dum in specz[found]]
        zprob = [np.array([dum]) for dum in zprob[found]]
        spectype = spectype[found]

        spectype[0] = -1   # case of no spectral type
        specz[0] = np.array([specz[0][0], 0.2])
        zprob[0] = np.array([0.6,0.4])

        spectype[1] = -1   # case of no spectral type
        specz[1] = np.array([specz[1][0], 0.8])
        zprob[1] = np.array([0.3,0.7])

        # observation['counts'] =observation['counts'][found]
        # observation['counts_cov'] =observation['counts_cov'][found]
        counts =[counts[i] for i in xrange(len(found)) if found[i]]
        counts_invcov = [counts_invcov[i] for i in xrange(len(found)) if found[i]]

        return {"specz": specz, "zprob": zprob, "spectype": spectype, "counts": counts, "counts_invcov": counts_invcov}