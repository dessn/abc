import logging
import numpy as np
from astropy.cosmology import FlatwCDM
import sncosmo
from astropy.table import Table


class Simulation(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_simulation(self, omega_m=0.3, H0=70, snIa_luminosity=-19.3, snIa_sigma=0.1,
                       num_transient=10, zmin=0.1, zmax=0.7, mean_num_obs=20, alpha=0.3, beta=4.0):
        np.random.seed(1)

        cosmology = FlatwCDM(Om0=omega_m, w0=-1, H0=H0)

        # Get redshift distribution of supernova
        tmin = 56700
        tmax = tmin + 500

        redshifts = np.linspace(zmin, zmax, num_transient).tolist()
        self.logger.info('Getting data for %d supernova' % len(redshifts))
        num_obs = np.ceil(np.random.normal(loc=mean_num_obs, scale=2.0, size=len(redshifts)))

        observations = [self.get_supernova(z, n, tmin, tmax, cosmology, alpha, beta,
                                           x0_mean=snIa_luminosity, x0_sigma=snIa_sigma)
                        for z, n, in zip(redshifts, num_obs)]
        lcs = [o[0] for o in observations]
        t0s = [o[1] for o in observations]
        x0s = [o[2] for o in observations]
        x1s = [o[3] for o in observations]
        cs = [o[4] for o in observations]
        t0sigma = [o[5] for o in observations]
        x0sigma = [o[6] for o in observations]
        x1sigma = [o[7] for o in observations]
        csigma = [o[8] for o in observations]
        theta = {
            r"$\Omega_m$": omega_m,
            r"$H_0$": H0,
            r"$M_0$": snIa_luminosity,
            r"$\sigma_{\rm SNIa}$": snIa_sigma,
            r"$\alpha$": alpha,
            r"$\beta$": beta
        }
        data = {"z": np.array(redshifts), "lcs": lcs,
                "x0": np.array(x0s), "x0s": np.array(x0sigma),
                "t0": np.array(t0s), "t0s": np.array(t0sigma),
                "x1": np.array(x1s), "c": np.array(cs),
                "x1s": np.array(x1sigma), "cs": np.array(csigma)}
        return data, theta

    def get_supernova(self, z, num_obs, tmin, tmax, cosmology, alpha, beta,
                      x0_mean=-19.3, x0_sigma=0.1):
        t0 = np.random.uniform(tmin, tmax)
        ts = np.linspace(t0 - 60, t0 + 60, num_obs)

        times = np.array([[t, t + 0.1, t + 0.2] for t in ts]).flatten()
        bands = [b for t in ts for b in ['desg', 'desr', 'desi']]
        gains = np.ones(times.shape)
        skynoise = 20 * np.ones(times.shape)
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
        x1 = np.random.normal(0., 1.)
        c = np.random.normal(0., 0.2)
        mabs = mabs - alpha * x1 + beta * c
        model.set_source_peakabsmag(mabs, 'bessellb', 'ab', cosmo=cosmology)

        x0 = model.get('x0')

        p = {'z': z,
             't0': t0,
             'x0': x0,
             'x1': x1,
             'c': c
             }

        lcs = sncosmo.realize_lcs(obs, model, [p])
        res, fitted_model = sncosmo.fit_lc(lcs[0], model, ['t0', 'x0', 'x1', 'c'])
        determined_parameters = {k: v for k, v in zip(res.param_names, res.parameters)}
        model.set(**determined_parameters)
        t0_ind = res.vparam_names.index('t0')
        x0_ind = res.vparam_names.index('x0')
        x1_ind = res.vparam_names.index('x1')
        c_ind = res.vparam_names.index('c')

        x0 = res.parameters[res.param_names.index('x0')]
        x1 = res.parameters[res.param_names.index('x1')]
        c = res.parameters[res.param_names.index('c')]
        t0 = res.parameters[res.param_names.index('t0')]

        sigma_x1 = np.sqrt(res.covariance[x1_ind, x1_ind])
        sigma_x0 = np.sqrt(res.covariance[x0_ind, x0_ind])
        sigma_c = np.sqrt(res.covariance[c_ind, c_ind])
        sigma_t0 = np.sqrt(res.covariance[t0_ind, t0_ind])
        return [lcs[0], t0, x0, x1, c, sigma_t0, sigma_x0, sigma_x1, sigma_c]


if __name__ == "__main__":
    sim = Simulation()


