import logging
import numpy as np
from astropy.cosmology import FlatwCDM
import sncosmo
from astropy.table import Table


class Simulation(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_simulation(self, omega_m=0.3, H0=70, snIa_luminosity=-19.3, snIa_sigma=0.1, snII_luminosity=-18.5,
                       snII_sigma=0.3, sn_rate=0.7, num_days=100, area=1, zmax=0.8, mean_num_obs=20):
        np.random.seed(1)

        # Get cosmology
        w_0 = -1
        cosmology = FlatwCDM(Om0=omega_m, w0=w_0, H0=H0)

        # Get redshift distribution of supernova
        tmin = 56700
        tmax = tmin + num_days
        model = sncosmo.Model(source='salt2')
        redshifts = list(sncosmo.zdist(0., zmax, time=num_days, area=area))
        types = ["Ia" if np.random.random() < sn_rate else "II" for z in redshifts]
        lums = [snIa_luminosity if t == "Ia" else snII_luminosity for t in types]
        sigs = [snIa_sigma if t == "Ia" else snII_sigma for t in types]

        self.logger.info('Getting data for %d days of transients, with %d supernova' % (num_days, len(redshifts)))

        num_obs = np.ceil(np.random.normal(loc=mean_num_obs, scale=2.0, size=len(redshifts)))

        observations = [self.get_supernova(z, n, tmin, tmax, cosmology, x0_mean=l, x0_sigma=s) for z, n, l, s in zip(redshifts, num_obs, lums, sigs)]

        theta = {
            "$w$": w_0,
            r"$\Omega_m$": omega_m,
            r"$H_0$": H0,
        }
        for i, obs in enumerate(observations):
            for k, v in obs.meta.items():
                key = "$%s_{%d}$" % (k, i)
                theta[key] = v
        print(theta)
        data = {"z_o": redshifts, "type_o": types, "lcs_o": observations}
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
             'x1': np.random.normal(0., 0.001*1.),
             'c': np.random.normal(0., 0.001*0.1)
             }

        lcs = sncosmo.realize_lcs(obs, model, [p])
        return lcs[0]


if __name__ == "__main__":
    sim = Simulation()


