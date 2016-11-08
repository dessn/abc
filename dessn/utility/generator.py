import numpy as np
import sncosmo
from astropy.table import Table
from astropy.cosmology import WMAP9

def get_obs_times_and_conditions(shallow=True, zp=None, num_obs=20, cadence=5, t0=1000,
                                 new_moon_t0=None, deltat=-30, weather=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if new_moon_t0 is None:
        new_moon_t0 = t0 - np.random.uniform(0, 29.5)
    ts = np.arange(t0 + deltat, (t0 + deltat) + cadence * num_obs, cadence)

    bands = [b for t in ts for b in ['desg', 'desr', 'desi', 'desz']]
    if zp is None:
        if shallow:
            zps = np.array([b for t in ts for b in [32.46, 32.28, 32.55, 33.12]])
        else:
            zps = np.array([b for t in ts for b in [34.24, 34.85, 34.94, 35.42]])
    else:
        zps = np.array([b for t in ts for b in zp])
    mins = np.array([b for t in ts for b in [22.1, 21.1, 20.1, 18.7]])
    maxs = np.array([b for t in ts for b in [19.4, 19.7, 19.4, 18.2]])
    times = np.array([[t, t + 0.05, t + 0.1, t + 0.2] for t in ts]).flatten()

    full = 0.5 + 0.5 * np.sin((times - new_moon_t0) * 2 * np.pi / 29.5)
    perm = np.random.uniform(-weather, weather, full.shape)
    seeing2 = np.random.uniform(4, 6, full.shape)
    sigma_psf = seeing2 / 2.36

    sky_noise = np.array(
        [np.sqrt(10.0 ** (((maxx - minn) * f + minn + p - zp) / -2.5) * 0.263 ** 2) *
         np.sqrt(4 * np.pi) * s
         for f, p, s, minn, maxx, zp in zip(full, perm, sigma_psf, mins, maxs, zps)])

    zpsys = ['ab'] * times.size
    gains = np.ones(times.shape)

    obs = Table({'time': times,
                 'band': bands,
                 'gain': gains,
                 'skynoise': sky_noise,
                 'zp': zps,
                 'zpsys': zpsys})

    return obs, t0, bands, zps


def generate_ia_light_curve(z, mabs, x1, c, get_calib=True, **kwargs):
    if "cosmo" in kwargs:
        cosmo = kwargs["cosmo"]
        del kwargs["cosmo"]
    else:
        cosmo = WMAP9

    model = sncosmo.Model(source='salt2-extended')
    model.set(z=z)
    model.set_source_peakabsmag(mabs, 'bessellb', 'ab', cosmo=cosmo)
    x0 = model.get('x0')
    seed = int(mabs + z * 100 + x1 * 10 + c * 10) % 4294967294

    obs, t0, bands, zps = get_obs_times_and_conditions(seed=seed, **kwargs)
    p = {'z': z, 't0': t0, 'x0': x0, 'x1': x1, 'c': c}
    np.random.seed(seed)
    lc = sncosmo.realize_lcs(obs, model, [p])[0]
    if get_calib:
        count = lc['flux'] / np.power(10, zps / 2.5)
        count_error = lc['fluxerr'] / np.power(10, zps / 2.5)
        filters = ['desg', 'desr', 'desi', 'desz']
        delta = 0.001
        lcs = [(None, 0, lc)]
        for f in filters:
            l = lc.copy()
            zp2 = zps + [delta if b == f else 0 for b in bands]
            l['flux'] = count * np.power(10, zp2 / 2.5)
            l['fluxerr'] = count_error * np.power(10, zp2 / 2.5)
            lcs.append((f, delta, l))
    else:
        lcs = [(None, 0, lc)]

    return lcs


def get_summary_stats(z, lc, method="emcee", convert_x0_to_mb=True):
    model = sncosmo.Model(source='salt2-extended')
    model.set(z=z)
    if method == "emcee":
        res, fitted_model = sncosmo.mcmc_lc(lc, model, ['t0', 'x0', 'x1', 'c'])
    elif method == "minuit":
        res, fitted_model = sncosmo.fit_lc(lc, model, ['t0', 'x0', 'x1', 'c'])
    else:
        raise ValueError("Method %s not recognised" % method)
    parameters = res.parameters[2:]

    if convert_x0_to_mb:
        determined_parameters = {k: v for k, v in zip(res.param_names, res.parameters)}
        model.set(**determined_parameters)
        mb = fitted_model.source.peakmag("bessellb", "ab")
        parameters = np.array([mb, parameters[1], parameters[2]])
        x0, x1, c = 1, 2, 3
        sigma_mb2 = 5 * np.sqrt(res.covariance[x0, x0]) / (2 * x0 * np.log(10))
        sigma_mbx1 = -5 * res.covariance[x0, x1] / (2 * x0 * np.log(10))
        sigma_mbc = -5 * res.covariance[x0, c] / (2 * x0 * np.log(10))
        cov = res.covariance
        cov = np.array([[sigma_mb2, sigma_mbx1, sigma_mbc],
                        [sigma_mbx1, cov[x1, x1], cov[x1, c]],
                        [sigma_mbc, cov[x1, c], cov[c, c]]])
    else:
        cov = res.covariance[1:, :][:, 1:]
    return parameters, cov


def check_lc_passes_cut(lc):
    """ Checks to see if the supplied light curve passes selection cuts."""
    bands = lc["band"]
    sn = lc["flux"] / lc["fluxerr"]
    sn_in_band = np.array([np.any(sn[bands == b] > 5.0) for b in np.unique(bands)])
    return sn_in_band.sum() >= 2


def get_ia_summary_stats(z, mabs, x1, c, method="minuit", **kwargs):
    lcs = generate_ia_light_curve(z, mabs, x1, c, **kwargs)
    if check_lc_passes_cut(lcs[0][-1]):
        results = [get_summary_stats(z, lc[-1], method=method) for lc in lcs]
        if len(results) == 1:
            return results[0][0], results[0][1], [None]*4, None
        base_param, base_cov = results[0]
        diffp = []
        diffc = []
        conv = 1  # Change to 1000 for millimag
        for result, lc in zip(results[1:], lcs[1:]):
            diffp.append((result[0] - base_param) / lc[1] / conv)
            diffc.append((result[1] - base_cov) / lc[1] / conv / conv)
            # print(lc[0], diffp[-1])
        return base_param, base_cov, np.array(diffp).T, np.array(diffc).T
    else:
        return None


def generate_ii_light_curve(z, mabs, source=None, **kwargs):
    if source is None:
        sources = [a[0] for a in sncosmo.builtins.models if a[1] == "SN IIP"]
        source = sources[np.random.randint(0, len(sources))]
    obs, t0, bands, zps = get_obs_times_and_conditions(**kwargs)
    model = sncosmo.Model(source=source)
    model.set(z=z)
    model.set_source_peakabsmag(mabs, 'bessellb', 'ab')
    p = {'z': z, 'amplitude': model.get('amplitude'), 't0': t0}
    lc = sncosmo.realize_lcs(obs, model, [p])[0]
    return lc
