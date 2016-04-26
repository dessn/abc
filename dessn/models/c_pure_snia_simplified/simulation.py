import logging
import numpy as np
from astropy.cosmology import FlatwCDM
import sncosmo
from astropy.table import Table


class Simulation(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_simulation(self, omega_m=0.3, H0=70, snIa_luminosity=-19.3, snIa_sigma=0.1,
                       num_transient=10, zmin=0.1, zmax=0.8, mean_num_obs=30):
        np.random.seed(1)

        cosmology = FlatwCDM(Om0=omega_m, w0=-1, H0=H0)

        # Get redshift distribution of supernova
        tmin = 56700
        tmax = tmin + 500

        redshifts = np.linspace(zmin, zmax, num_transient).tolist()
        self.logger.info('Getting data for %d supernova' % len(redshifts))
        num_obs = np.ceil(np.random.normal(loc=mean_num_obs, scale=2.0, size=len(redshifts)))

        observations = [self.get_supernova(z, n, tmin, tmax, cosmology,
                                           x0_mean=snIa_luminosity, x0_sigma=snIa_sigma)
                        for z, n, in zip(redshifts, num_obs)]
        mbs = [o[0] for o in observations]
        x1s = [o[1] for o in observations]
        cs = [o[2] for o in observations]
        ics = [o[3] for o in observations]
        theta = {
            r"$\Omega_m$": omega_m,
            r"$H_0$": H0,
            r"$L_{\rm SnIa}$": snIa_luminosity,
            r"$\sigma_{\rm SnIa}$": snIa_sigma,
        }
        for i, (mb, x1, c) in enumerate(zip(mbs, x1s, cs)):
            theta["$m_{B %d}$" % i] = mb
            theta["$x_{1 %d}$" % i] = x1
            theta["$c{ %d}$" % i] = c
        print(theta)
        data = {"z_o": redshifts, "mbs": mbs, "x1s": x1s, "cs": cs, "icovs": ics}
        return data, theta

    def get_supernova(self, z, num_obs, tmin, tmax, cosmology, x0_mean=-19.3, x0_sigma=0.1):
        t0 = np.random.uniform(tmin, tmax)
        ts = np.linspace(t0 - 60, t0 + 60, num_obs)

        times = np.array([[t, t + 0.1, t + 0.2] for t in ts]).flatten()
        bands = [b for t in ts for b in ['desg', 'desr', 'desi']]
        gains = np.ones(times.shape)
        skynoise = 50 * np.ones(times.shape)
        zp = 30 * np.ones(times.shape)
        zpsys = ['ab'] * times.size

        obs = Table({'time': times,
                     'band': bands,
                     'gain': gains,
                     'skynoise': skynoise,
                     'zp': zp,
                     'zpsys': zpsys})

        model = sncosmo.Model(source='salt2')

        mabs = np.random.normal(x0_mean, x0_sigma)
        model.set(z=z)
        model.set_source_peakabsmag(mabs, 'bessellb', 'ab', cosmo=cosmology)
        x0 = model.get('x0')
        p = {'z': z,
             't0': t0,
             'x0': x0,
             'x1': np.random.normal(0., 1.),
             'c': np.random.normal(0., 0.1)
             }

        lcs = sncosmo.realize_lcs(obs, model, [p])
        res, fitted_model = sncosmo.fit_lc(lcs[0], model, ['t0', 'x0', 'x1', 'c'])
        determined_parameters = {k: v for k, v in zip(res.param_names, res.parameters)}
        model.set(**determined_parameters)
        mb = model.bandmag("bessellb", "ab", t0)
        x0_ind = res.vparam_names.index('x0')
        x1_ind = res.vparam_names.index('x1')
        c_ind = res.vparam_names.index('c')

        x0 = res.parameters[x0_ind]
        x1 = res.parameters[x1_ind]
        c = res.parameters[c_ind]
        sigma_mb2 = (5.0*res.errors['x0']/x0 * mb)**2 #TODO: FIND THIS PROPERLY
        sigma_mbx1 = -5 * res.covariance[x0_ind, x1_ind] / (2 * x0 * np.log(10))
        sigma_mbc = -5 * res.covariance[x0_ind, c_ind] / (2 * x0 * np.log(10))

        covariance = np.array([[sigma_mb2, sigma_mbx1, sigma_mbc],
                              [sigma_mbx1, res.covariance[x1_ind, x1_ind], res.covariance[x1_ind, c_ind]],
                              [sigma_mbc, res.covariance[x1_ind, c_ind], res.covariance[c_ind, c_ind]]])
        icov = np.linalg.inv(covariance)
        return [mb, x1, c, icov]


if __name__ == "__main__":
    sim = Simulation()


