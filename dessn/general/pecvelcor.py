import numpy as np
import matplotlib.pyplot as plt

'''
Functions to derive the uncertainty associated with correcting observed redshifts in the CMB frame
for peculiar motion. Which function(s) is/are needed depends on the approach of the cosmological fit.

The simplest solution is to propagate the redshift uncertainty from get_sigma_redshift_pecvel() as
the uncertainty is constant in redshift. If that is not possible the uncertainty can be shifted
to the distance modulus (or whatever magnitude you have in your code, possibly m_B). For magnitudes
however the uncertainty is not constant, but decreases with redshift (corresponding to the slope of
the mu vs redshift plot). The function to call is then get_sigma_mu_pecvel() which takes as input a
numpy like array that contains the cosmological redshifts of the SNe (z_Hd or zcor) and returns a
numpy array with the magnitude uncertainties for each redshift to propagate. If you are working
with the full covariance matrix this array is added to the diagonal of that covariance matrix.

IMPORTANT: If you are already propagating an uncertainty from the linear theory parameter beta,
which scales linearly with the peculiar velocites, set sigma_lineartheory to zero.

For further details see: https://www.overleaf.com/read/bzpnwnmcqhfj

'''


def get_sigma_redshift_pecvel(sigma_lineartheory=150., sigma_shotnoise=125., sigma_missingdata=100.):
    '''Return the redshift uncertainty asociated with correcting redhift for
    peculiar motion.

    Keyword arguments:
    sigma_lineartheory -- uncertainty from limits of linear theory (default 150.0)
    sigma_shotnoise -- uncertainty from shotnoise in 2M++ (default 125.0)
    sigma_missingdata -- uncertainty from missing data in 2M++ (default 100.0)
    '''

    assert isinstance(sigma_lineartheory, float), 'Expected float but found %s' % type(sigma_lineartheory)
    assert isinstance(sigma_shotnoise, float), 'Expected float but found %s' % type(sigma_lineartheory)
    assert isinstance(sigma_missingdata, float), 'Expected float but found %s' % type(sigma_lineartheory)

    sigma_redshift_pecvel = np.sqrt(sigma_lineartheory ** 2 + sigma_shotnoise ** 2 + sigma_missingdata ** 2) / 2.99792e5

    return sigma_redshift_pecvel


def get_sigma_mu_pecvel(redshifts, sigma_lineartheory=150., sigma_shotnoise=125., sigma_missingdata=100.):
    '''Return the magnitude uncertainty asociated with correcting redhift for
	peculiar motion.

	Keyword arguments:
	redshifts -- array of cosmological redshifts in CMB frame
	sigma_lineartheory -- uncertainty from limits of linear theory (default 150.0)
	sigma_shotnoise -- uncertainty from shotnoise in 2M++ (default 125.0)
	sigma_missingdata -- uncertainty from missing data in 2M++ (default 100.0)
	'''

    assert isinstance(redshifts, np.ndarray), 'Expected numpy array but found %s' % type(redshifts)
    assert np.min(redshifts) > 0., 'Zero or negative redshift in input array!'

    sigma_redshift_pecvel = get_sigma_redshift_pecvel(sigma_lineartheory=sigma_lineartheory,
                                                      sigma_shotnoise=sigma_shotnoise,
                                                      sigma_missingdata=sigma_missingdata)

    sigma_mu_pecvel = sigma_redshift_pecvel * 5. / np.log(10.) * (1. + redshifts) ** 2 / (
        redshifts * (1 + 0.5 * redshifts))

    return sigma_mu_pecvel


if __name__ == '__main__':

    # Some very basic tests


    # Type test
    try:
        sigma_redshift_pecvel = get_sigma_redshift_pecvel()
        if type(sigma_redshift_pecvel) == np.float64:
            pass
        else:
            print("get_sigma_redshift_pecvel() type test failed, did not return a np.float64!")
    except:
        pass

        # Assertion test
    try:
        sigma_redshift_pecvel = get_sigma_redshift_pecvel(sigma_lineartheory=[])
        if type(sigma_redshift_pecvel) == np.float64:
            pass
        else:
            print("get_sigma_redshift_pecvel() type test failed, did not return a np.float64!")
    except AssertionError:
        pass
    except:
        "get_sigma_redshift_pecvel() assertion test failed, did not raise AssertionError!"

    # Assertion test
    try:
        sigma_redshift_pecvel = get_sigma_redshift_pecvel(sigma_missingdata=[])
        if type(sigma_redshift_pecvel) == np.float64:
            pass
        else:
            print("get_sigma_redshift_pecvel() type test failed, did not return a np.float64!")
    except AssertionError:
        pass
    except:
        "get_sigma_redshift_pecvel() assertion test failed, did not raise AssertionError!"

    # Assertion test
    try:
        sigma_redshift_pecvel = get_sigma_redshift_pecvel(sigma_lineartheory=[])
        if type(sigma_redshift_pecvel) == np.float64:
            pass
        else:
            print("get_sigma_redshift_pecvel() type test failed, did not return a np.float64!")
    except AssertionError:
        pass
    except:
        "get_sigma_redshift_pecvel() assertion test failed, did not raise AssertionError!"

    ### Function: get_sigma_mu_pecvel()

    # Output type test
    try:
        if type(get_sigma_mu_pecvel(np.array([0.1]))) == np.ndarray:
            pass
        else:
            print("get_sigma_mu_pecvel() type test failed, did not return numpy array!")
    except:
        pass

    # Assertion test
    try:
        get_sigma_mu_pecvel(np.array([0.1, -0.1, 2.]))
    except AssertionError:
        pass
    except:
        "get_sigma_mu_pecvel() assertion test failed, did not raise AssertionError!"

    # Assertion test
    try:
        get_sigma_mu_pecvel(np.array([0.1, 0.1, 2., 0.]))
    except AssertionError:
        pass
    except:
        "get_sigma_mu_pecvel() assertion test failed, did not raise AssertionError!"

    # Assertion test
    try:
        get_sigma_mu_pecvel(1.)
    except AssertionError:
        pass
    except:
        "get_sigma_mu_pecvel() assertion test failed, did not raise AssertionError!"

    # Assertion test
    try:
        get_sigma_mu_pecvel([0.1, 0.2, 0.3])
    except AssertionError:
        pass
    except:
        "get_sigma_mu_pecvel() assertion test failed, did not raise AssertionError!"

    # This produces a plot to visually test the output of get_sigma_mu_pecvel()
    redshifts = np.linspace(0.001, 0.1, 100)

    sigma_mu_pecvel = get_sigma_mu_pecvel(redshifts)

    plt.figure()
    plt.xlim((0., .1))
    plt.ylim((0, 0.2))
    plt.xlabel('Cosmological redshift', size='large')
    plt.ylabel(r'$\sigma_{\mu ,pec}$', size='x-large')
    plt.plot([0.01, 0.05, 0.1], [.1612858744, 0.03418254355, 0.01831115415], 'kx')
    plt.plot(redshifts, sigma_mu_pecvel, 'k')
    plt.plot(redshifts, get_sigma_mu_pecvel(redshifts, sigma_lineartheory=300.), 'b')
    plt.plot([0.02, 0.05, 0.1], [.1268554479, 0.05298401655, 0.02838286428], 'bx')
    plt.text(0.02, 0.175,
             'Visual test: Black crosses should intersect the black line, \n ditto with blue crosses and line')
    plt.show()
