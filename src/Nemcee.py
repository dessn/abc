#!/usr/bin/env python
import sys
import numpy
import scipy.special
import scipy.integrate
import emcee
from astropy.cosmology import FlatwCDM
from astropy import constants as const
from astropy import units as u

from emcee.utils import MPIPool

# from pymc3 import Model, Normal, Lognormal, Uniform
# from pymc3.distributions import Continuous
# import theano.tensor as T



ln10_2p5 = numpy.log(10)/2.5
magtoflux  = numpy.exp(ln10_2p5)


class Inputs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

inputs = Inputs(Om0 = 0.28, w0=-1., rate_II_r=2., logL_snIa=numpy.log(1.), sigma_snIa=0.01, logL_snII = numpy.log(0.5), sigma_snII=0.4, Z=0.)
uncertainties = Inputs(logL_snIa=0.01, sigma_snIa=0.1, logL_snII = 0.1, sigma_snII=0.1, Z=0.01)

fluxthreshold = 0.4e-8

def luminosity_distance(z, Om0, w0):
    cosmo = FlatwCDM(H0=72, Om0=Om0, w0=w0)
    return cosmo.luminosity_distance(z).value

def normalization_integrand(n, lnL0, sigma_lnL, Z, threshold, ld, dcounts):
    #   The L value being sampled:
    #   lnL = n*sigma_lnL+lnL0
    #   exp(lnL) = exp(n)**sigma_lnL * exp(lnL0)
    #
    #   The probability of that lnL proportional to
    #   1/sigma_lnL * exp(-(n**2 / 2))
    #
    #   The counts corresponding to that lnL
    #   L = exp(lnL)
    #   flux = 1/4/pi L/ld**2
    #   counts = flux 10**(Z/2.5)
    #   counts = flux [exp(ln10/2.5)]**Z
    #   counts = 1/4/pi exp(lnL)/ld**2 [exp(ln10/2.5)]**Z
    #   counts = 1/4/pi exp(n)**sigma_lnL * exp(lnL0)/ld**2 [exp(ln10/2.5)]**Z

    # the gaussian on logL (leaving out the sqrt 2pi)

    firstterm  = numpy.exp(-n**2/2) #the sigma_lnL here cancels with the normalization factor
    L = numpy.exp(lnL0+n*sigma_lnL)
    flux = L/4/numpy.pi/ld**2
    counts = flux * 10**(Z/2.5)
    ccdf  = 0.5 * (1- scipy.special.erf((threshold-counts)/ numpy.sqrt(2.)/dcounts))
    return firstterm*ccdf

def lognormal_logp(value, mu=None, tau=None):
    return -0.5 * tau * (numpy.log(value) - mu)**2 + 0.5 * numpy.log(tau/(2. * numpy.pi))   \
        - numpy.log(value)

def normal_logp(value, mu=None, tau=None):
    return (-tau * (value - mu)**2 + numpy.log(tau / numpy.pi / 2.)) / 2

def LogLuminosityMarginalizedOverType_logp(value, mus=None, taus=None, p=None):
    r"""The distribution for the luminosity marginalized over two kinds
    of astronomical sources:

    .. math::
        pdf(Luminosity | Type prob, logL_snIa, logL_snII)
            = sum_i pdf(Luminosity | Type_i, logL_snIa, logL_snII) *
                pdf(Type_i | Type prob)


    This class should be generalized to handle multiple types each with its own model

    Count does not depend explicitly on Type, so the summation over type is done here

    Parameters
    -----------
mu=    tau=mus : array
        the logL_X
    sds : array
        the sigma_X
    p : theano.tensor
        The probability of the first case
        pdf(T1|X), as a consequence pdf(T2|X) = 1-p
    """
    logterms = numpy.array([normal_logp(value, mus[0], taus[0]),normal_logp(value, mus[1], taus[1])])
    logterm_max = numpy.max(logterms)
    logterms = numpy.array([p, 1-p]) * numpy.exp(logterms-logterm_max)

    # print 'LLOT',  logterm_max + numpy.log(logterms.sum())
    return logterm_max + numpy.log(logterms.sum())

    # return numpy.log(p * numpy.exp(normal_logp(value, mus[0], taus[0])) + \
    #     (1-p) * numpy.exp( normal_logp(value, mus[1], taus[1])))

def LogLuminosityGivenSpectype_logp(value, mu=None, tau=None, p=None):
    r"""The distribution for the joint spectype and log-luminosity.

    The input variable ls log-luminosity, not luminosity.  This therefore subclasses
    Normal.

    It is the product of the probability of the type times the pdf of the luminosity
    which in this case is Lognormal (the Parent class).  Do templates exist in Python?

    .. math::
        pdf(Obs type, Luminosity | Type prob, logL_snIa, logL_snII)
            = sum_i pdf(Obs type| Type_i) *
                pdf(Luminosity | Type_i, logL_snIa, logL_snII) *
                pdf(Type_i | Type prob)
            = pdf(Luminosity | Type=Obs type, logL_snIa, logL_snII) *
                pdf(Type=Obs type | Type prob)

    This class should be generalized to handle multiple types

        
    Parameters
    -----------
    p : Theano.Tensor
        pdf(T|X)
    """
    # print 'LLGST', numpy.log(p) + normal_logp(value, mu, tau)
    return numpy.log(p) + normal_logp(value, mu, tau)

def Counts_logp(value, fluxes=None, pzs = None, Z=None, csigma=None):
    r"""The distribution for the joint spectype and log-luminosity

    pdf of counts given a threshold

    .. math::
        pdf(observed redshift, Counts | Luminosity, Redshift, Cosmology, Calibration)
            = sum_i p_i pdf(Counts | Flux_i, Redshift=observer_redshift_i, Calibration)
        
    Parameters
    -----------
    threshold : theano.tensor
        minimum counts in order to be discovered

    """
    ans=0.
    logterms = []
    for index in xrange(len(pzs)):
        flux = fluxes[index]
        counts  = flux*magtoflux**Z
        tau = csigma**(-2)
        logterms.append(normal_logp(value, counts, tau))
    logterms = numpy.array(logterms)
    logterm_max = numpy.max(logterms)
    logterms = pzs * numpy.exp(logterms-logterm_max)
    # print 'C',logterm_max + numpy.log(logterms.sum())
    return logterm_max + numpy.log(logterms.sum())

    #ans = ans + pzs[index] *numpy.exp(normal_logp(value, counts, tau))

    #need to add threshold

    #return numpy.log(ans)

def SampleRenormalization_logp(threshold = 0, logL_snIa=None, sigma_snIa=None, logL_snII=None, sigma_snII=None,\
            luminosity_distances=None, Z=None, pzs=None, prob=None, dcounts=None):
    r""" Renormalization factor from sample selection for a typed supernova with fixed
    redshift.

    .. math:
        P(S_c, S_T| z=z_o, X) =
            sum_i  P(T_i|z=z_o, X)
                \int dL p(L|T_i, z=z_o, X)  \left[\int_{c_T}^{\infty} dc_o  p(c_o | T=T_i, L, z=z_o, X)\right]


    """
    for ld, pz, indz in zip(luminosity_distances,pzs,xrange(len(pzs))):
        #sum over type
        tsum = prob*scipy.integrate.quad(normalization_integrand, -5., 5., args=(logL_snIa, sigma_snIa, Z, threshold, ld, dcounts))[0] + \
            (1-prob)*scipy.integrate.quad(normalization_integrand, -5., 5., args=(logL_snII, sigma_snII, Z, threshold, ld, dcounts))[0]
        if indz == 0:
            ans = pz*tsum
        else: 
            ans = ans + pz*tsum
#    print 'SR',-numpy.log(ans)
    if ans ==0.:
    	return -numpy.inf
    return -numpy.log(ans)

def pgm():
    from daft import PGM, Node, Plate
    from matplotlib import rc
    rc("font", family="serif", size=8)
    rc("text", usetex=True)

    pgm = PGM([9.5, 8.5], origin=[0., 0.2], observed_style='inner')

    #pgm.add_node(Node('dispersion',r"\center{$\sigma_{Ia}$ \newline $\sigma_{!Ia}$}", 1,6,scale=1.2,aspect=1.8))
    pgm.add_node(Node('Rate_Ia',r"{SNIa Rate}", 1,8, fixed=1))
    pgm.add_node(Node('Rate_II',r"{SNII Rate}", 2,8,scale=1.6,aspect=1.2))
    pgm.add_node(Node('L_Ia',r"{SNIa L, $\sigma_L$}", 3,8,scale=1.6,aspect=1.2))
    pgm.add_node(Node('L_II',r"{SNII L, $\sigma_L$}", 4,8,scale=1.6,aspect=1.2))
    pgm.add_node(Node('Cosmology',r"Cosmology", 7,8, scale=1.6,aspect=1.2))
    pgm.add_node(Node('Calibration',r"Calibration", 8, 8, scale=1.6,aspect=1.2))

 #   pgm.add_node(Node('Neighbors',r"\centering{Neighbor \newline Redshifts}", 5,7, scale=1.6,aspect=1.2))
    pgm.add_node(Node('Redshift',r"{Redshift}", 6,7, scale=1.6,aspect=1.2))

    pgm.add_node(Node('Type_prob',r"Type prob", 1,6, fixed=1,offset=(20,-10)))
    pgm.add_node(Node('Distance',r"$L_D$", 7,6, fixed=1,offset=(10,10)))

    pgm.add_node(Node('Type',r"Type", 1, 5, scale=1.6,aspect=1.2))

    pgm.add_node(Node('Luminosity',r"Luminosity", 4, 4, scale=1.6,aspect=1.2))
    pgm.add_node(Node('Flux',r"Flux", 7, 3, scale=1.2,fixed=True,offset=(-20,-20)))


    pgm.add_node(Node('Obs_Type',r"Obs type", 1, 1, scale=1.6,aspect=1.2,observed=1))
    pgm.add_node(Node('Obs_Redshift',r"Obs redshift", 6, 1, scale=1.6,aspect=1.2,observed=1))
    pgm.add_node(Node('Counts',r"Counts", 8, 1, scale=1.2,observed=1))


    pgm.add_edge("Rate_Ia","Type_prob")
    pgm.add_edge("Rate_II","Type_prob")

    pgm.add_edge("Cosmology","Distance")
    pgm.add_edge("Redshift","Distance")

    pgm.add_edge("Type_prob", "Type")

    pgm.add_edge("Type","Luminosity")
    pgm.add_edge("L_Ia", "Luminosity")
    pgm.add_edge("L_II", "Luminosity")

    pgm.add_edge("Luminosity","Flux")
    pgm.add_edge("Redshift","Flux")
    pgm.add_edge("Distance","Flux")

    pgm.add_edge("Type","Obs_Type")
#    pgm.add_edge("Neighbors","Obs_Redshift")
    pgm.add_edge("Redshift","Obs_Redshift")

    pgm.add_edge("Flux","Counts")
    pgm.add_edge("Calibration","Counts")

    # Big Plate: Galaxy
    pgm.add_plate(Plate([0.4, 0.5, 8.2, 7.],
                        label=r"SNe $i = 1, \cdots, N_{SN}$",
                        shift=-0.2,label_offset=[20,2]))

    pgm.add_plate(Plate([0.5, 3.5, 4., 2.],
                        label=r"Type $\in \{Ia, II\}$",
                        shift=-0.2,label_offset=[20,2]))
    # Render and save.

    pgm.render()

    # pgm.figure.text(0.01,0.9,r'\underline{UNIVERSAL}',size='large')
    # pgm.figure.text(0.01,0.55,r'{\centering \underline{INDIVIDUAL} \newline \underline{SN}}',size='large')
    # pgm.figure.text(0.01,0.2,r'\underline{OBSERVATORY}',size='large')
    # pgm.figure.text(0.01,0.1,r'\underline{DATA}',size='large')


    pgm.figure.savefig("../results/nodes_pgm.pdf")


def simulateData():


    # the number of transients
    nTrans = 30

    # set the state of the random number generator
    seed=0
    numpy.random.seed(seed)

    # simulated data in the dictionary observation, including photometry at peak,
    # spectroscopic redshift, and spectroscopic type.
    # the convention is SNIa are '0', SNII are '1'
    # the current implementation is barebones

    observation=dict()
    observation['specz'] = numpy.random.uniform(low=0.1, high=0.8, size=nTrans)
    observation['zprob'] = numpy.zeros(nTrans)+1.
    spectype = numpy.random.uniform(low=0, high=1, size=nTrans)
    snIarate = 1./(1+inputs.rate_II_r)
    observation['spectype'] = numpy.zeros(nTrans,dtype=int)
    observation['spectype'][spectype > snIarate]=1

    luminosity = (1.-observation['spectype'])*numpy.exp(inputs.logL_snIa)*10**(numpy.random.normal(0, inputs.sigma_snIa/2.5, size=nTrans)) \
        + observation['spectype']*numpy.exp(inputs.logL_snII)*10**(numpy.random.normal(0, inputs.sigma_snII/2.5,size=nTrans))
    cosmo = FlatwCDM(H0=72, Om0=inputs.Om0, w0=inputs.w0)
    ld = cosmo.luminosity_distance(observation['specz']).value
    # h0 = (const.c/cosmo.H0).to(u.Mpc).value


    observation['counts'] = luminosity / 4/numpy.pi/ld/ld*10**(inputs.Z/2.5)

    # plt.scatter(observation['specz'],-2.5*numpy.log10(observation['counts']))

    found  = observation['counts'] >= fluxthreshold
    nTrans =  found.sum()
    observation['specz'] = numpy.reshape(observation['specz'][found],(nTrans,1))
    observation['zprob'] = numpy.reshape(observation['zprob'][found],(nTrans,1))
    observation['spectype'] =observation['spectype'][found]
#    observation['spectype'][0] = -1   # case of no spectral type
    observation['counts'] =observation['counts'][found]
    return observation

def lnprob(theta, co, zo, dco, pzo, spectypeo):

    Om0, w0, rate_II_r, logL_snIa, sigma_snIa, logL_snII, sigma_snII, Z = theta[0:8]
    lnLs = theta[8:]

    # basic_model = Model()

    # with basic_model:
    ans = 0.
    r"""
    Cosmology Node.

    The FlatwCDM cosmology.  

    pdf(Om0, w0)

    We need the flexibility to switch in and out different cosmological models.  The function
    that describes luminosity distance is specific to the model: the parameters and function
    should be packaged together.

    Parameters
    ----------
    Om0:    Omega_M ~ lognormal(0.28, 0.1)
    w0:     constant equation of state w ~ normal(-1,-0.05)
    """

    # _Om0 = Lognormal('Om0', mu=numpy.log(0.28), tau=1/.1/.1, observed = Om0)
    # _w0 = Normal('w0', mu=-1, sd=0.05, observed = w0)
    if Om0 < 0 or Om0 >1:
        return -numpy.inf

    #ans = lognormal_logp(Om0, numpy.log(0.28), tau=0.1**(-2))

    if w0< -1.5 or w0>=-0.1:
        return -numpy.inf
    #ans += normal_logp(w0, 1, 0.05**(-2))

    """
    Calibration Node.

    Global zeropoints for each band.

    pdf(Z)

    The transmission function of the bands will be used later.  The transmission and zeropoints
    should be packaged together. More complicated parameterizations of calibration are expected.

    Parameters
    -----------
    Z:  zeropoint (in mag) for the bands ~ normal(0,0.02)

    """
    # _Z = Normal('zeropoints', mu=0, sd=.02, observed=Z)
    ans += normal_logp(Z, inputs.Z, (ln10_2p5*uncertainties.Z)**(-2))

    """
    SN Ia Rate Node.  

    rate_Ia_r = constant

    For SN cosmology the relative rates between different populations are sufficient.  Rates of
    all types are relative the snIa rate, so snIa rate is taken to be 1.

    Parameters
    -----------
    rate_Ia_r =1    : the relative rates are relative to type Ia. Fixed.

    """

    rate_Ia_r = 1.


    """
    SN II Rate Node.

    The rate of SNe II realtiave SNIa.

    pdf(rate_II_r)

    Along with the rate parameters is a rate model.

    There should be equivalent nodes for all other transient types being modeled.

    Parameters
    ----------

    rate_II_r     : relative rate of SNe II compared to SNe Ia. 

    """
    if rate_II_r<0:
        return -numpy.inf

    # _rate_II_r = Uniform('rate_II_r', lower=0.25, upper=4, observed= rate_II_r)
    # ans += constant 

    """
    SN Ia luminosity Node.  (actually working in log-L)

    pdf(logL_snIa, sigma_snIa)

    For the moment consider the SN to be phase-indepemdent with no internal parameters.  Eventually
    this will represent time-evolving SED, e.g. SALT2.


    Parameters
    ----------

    logL_snIa   :       SN Ia mean log-luminosity   ~normal
    sigma_snIa :        intrinsic dispersion (mag)  ~lognormal

    """
    # _logL_snIa = Normal('logL_snIa', mu=numpy.log(1), sd = 0.02, observed = logL_snIa)
    # _sigma_snIa = Lognormal('sigma_snIa', mu=numpy.log(0.1), tau=1./0.1/0.1, observed = sigma_snIa)

    ans+= normal_logp(logL_snIa,  inputs.logL_snIa,  (ln10_2p5*uncertainties.logL_snIa)**(-2))

    if sigma_snIa < 0:
        return -numpy.inf
    ans+= lognormal_logp(sigma_snIa, numpy.log(inputs.sigma_snIa), (ln10_2p5*uncertainties.sigma_snIa)**(-2))

    """
    SN Ia luminosity Node.  (actually working in log-L)

    pdf(logL_snII, sigma_snIa)

    Parameters
    ----------

    logL_snII   :       SN II mean log-luminosity
    sigma_snII :        intrinsic dispersion (mag)

    """
    # _logL_snII = Normal('logL_snII', mu=numpy.log(0.5), sd=0.02, observed  = logL_snII)
    # _sigma_snII = Lognormal('sigma_snII', mu=numpy.log(0.4), tau=1./0.1/0.1, observed = sigma_snII)

    ans+= normal_logp(logL_snII, inputs.logL_snII, (ln10_2p5*uncertainties.logL_snII)**(-2))
    if sigma_snII < 0:
        return -numpy.inf
    ans+= lognormal_logp(sigma_snII, numpy.log(inputs.sigma_snII), (ln10_2p5*uncertainties.sigma_snII)**(-2))

    """
    Enter the plate that considers one supernova at a time
    """

    for lnL, counts, zs, dcounts, pzs, spectype in zip(lnLs, co, zo, dco, pzo, spectypeo):

        """
        Type Probability Node.

        Probabilities of being a type of object.  For now only SN Ia, and SN II.

        Dependencies
        -------------

        rate_Ia_r   :   Type Ia rate
        rate_II_r   :   Type II rate
        host galaxy :   Not implemented now but eventually depends on host properties

        Parameters
        ----------

        prob :          probability of the object being a type Ia.  Fixed.
        """

        prob = rate_Ia_r/(rate_Ia_r+rate_II_r)


        """
        Type Node.

        Not explicitly considered in our model.
        """

        """
        Observed Type Node and Luminosity Node.

        pdf(Obs type, Luminosity | Type prob, logL_snIa, logL_snII)

        There are two possibilities:

        1. There is an observed type assumed to be perfect.

            pdf(Obs type | Type) = delta(Obs type - Type)

            then 
            
            pdf(Obs type, Luminosity | Type prob, logL_snIa, logL_snII)
                = sum_i pdf(Obs type| Type_i) *
                    pdf(Luminosity | Type_i, logL_snIa, logL_snII) *
                    pdf(Type_i | Type prob)
                = pdf(Luminosity | Type=Obs type, logL_snIa, logL_snII) *
                    pdf(Type=Obs type | Type prob)

            The class LogLuminosityGivenSpectype is responsible for providing this pdf

        2. There is no observed type.

            pdf(Luminosity | Type prob, logL_snIa, logL_snII)
                = sum_i pdf(Luminosity | Type_i, logL_snIa, logL_snII) *
                    pdf(Type_i | Type prob)

            The class LuminosityMarginalizedOverType is responsible for providing this pdf

        Dependencies
        ------------

        prob        :
        logL_snIa   :
        sigma_snIa  :
        logL_snII   :
        sigma_snII  :

        Parameters
        ----------

        obstype         :   observed type, SN Ia=0, SNII=1 Marginalized over
        Luminosity      :

        """
        if spectype == -1 :
            # logluminosity = LogLuminosityMarginalizedOverType('logluminosity'+str(i), 
            #     mus=[logL_snIa, logL_snII], \
            #     sds = [numpy.log(10)/2.5*sigma_snIa,numpy.log(10)/2.5*sigma_snII], p=prob, \
            #     testval = 1., observed  = lnL)
            ans += LogLuminosityMarginalizedOverType_logp(lnL, mus=[logL_snIa, logL_snII], \
                taus = numpy.power([ln10_2p5*sigma_snIa,ln10_2p5*sigma_snII],-2), p=prob)
        else:
            if spectype == 0:
                usemu = logL_snIa
                usesd = ln10_2p5*sigma_snIa
                usep = prob
            else:
                usemu = logL_snII
                usesd = ln10_2p5*sigma_snII
                usep = 1-prob

            # logluminosity = LogLuminosityGivenSpectype('logluminosity'+str(i), \
            #         mu=usemu,sd=usesd, p=usep, observed = lnL)
            ans+= LogLuminosityGivenSpectype_logp(lnL, mu=usemu, tau=usesd**(-2), p=usep)
            
        #luminosity = numpy.exp(logluminosity)

        luminosity = numpy.exp(lnL)

        """
        Redshift Node.

        Not considered explicitly in our model.

        """

        """
        Observed Redshift, Counts Node.

        pdf(observed redshift, Counts | Luminosity, Redshift, Cosmology, Calibration)
            = pdf(observed redshift| Redshift) *
                pdf(Counts | Luminosity, Redshift, Cosmology, Calibration)

        The pdf of the observed redshift is assumed to be a sum of delta functions, perfectly
        measured redshift of the supernova or redshifts of potential galaxy hosts.

        pdf(observed redshift | Redshift) = sum_i p_i delta(observer redshift_i - Redshift)

        where p_i is the probability of observer redshift_i being the correct redshift.

        so

        pdf(observed redshift, Counts | Luminosity, Redshift, Cosmology, Calibration)
            = sum_i p_i pdf(Counts | Luminosity, Redshift=observer_redshift_i, Cosmology, Calibration)

        The class CountsWithThreshold handles this pdf

        Dependencies
        ------------

        luminosity  :   luminosity
        redshift    :   host redshift
        cosmology   :   cosmology
        Calibration :   calibration

        Parameters
        -----------

        observed_redshift   Marginalized over
        counts

        """

        lds=[]
        fluxes=[]
        for z_ in zs:
            # ld = 0.5/h0*(z_+T.sqr(z_))* \
            #     (1+ 1//T.sqrt((1+z_)**3 * (Om0 + (1-Om0)*(1+z_)**(3*w0))))
            ld = luminosity_distance(z_, Om0, w0)
            lds.append(ld)
            fluxes.append(luminosity/4/numpy.pi/ld**2)

        # c = Counts('counts'+str(i),fluxes =fluxes,  \
        #     pzs = pzs, Z=Z, csigma = dcounts, observed  =counts)

        ans+= Counts_logp(counts, fluxes =fluxes, pzs = pzs, Z=Z, csigma = dcounts)

        # normalization=SampleRenormalization('normalization'+str(i), threshold = 1e-9, 
        #     logL_snIa=logL_snIa, sigma_snIa=sigma_snIa, logL_snII=logL_snII, sigma_snII=sigma_snII,
        #     luminosity_distances=lds, Z=Z, pzs=pzs, prob=prob, observed=1)

        ans += SampleRenormalization_logp(threshold = fluxthreshold, \
            logL_snIa=logL_snIa, sigma_snIa=sigma_snIa, logL_snII=logL_snII, sigma_snII=sigma_snII,\
            luminosity_distances=lds, Z=Z, pzs=pzs, prob=prob, dcounts=dcounts)

   # print 'Done', ans
    return ans

import pickle
import corner
import matplotlib.pyplot as plt

def runModel():

    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    # pool=None

    observation = simulateData()
    nTrans = len(observation['spectype'])

    ndim, nwalkers = 8+ nTrans, 1000

    # mns = numpy.concatenate(([inputs.Om0, inputs.w0, inputs.rate_II_r, inputs.logL_snIa, inputs.sigma_snIa, \
    #             inputs.logL_snII,inputs.sigma_snII,inputs.Z], -.35*numpy.zeros(nTrans)))
    sigs = numpy.concatenate(([.1,.2,.1, uncertainties.logL_snIa, uncertainties.sigma_snIa, uncertainties.logL_snII, uncertainties.sigma_snII, uncertainties.Z], .05+numpy.zeros(nTrans)))

    p0=[]
    for i in range(nwalkers):
        dum = numpy.random.rand(nTrans)
        dum = numpy.array(numpy.round(dum),dtype='int')
        lnL_init = dum + (1-dum)*0.5
        lnL_init = numpy.log(lnL_init)

        mns = numpy.concatenate(([inputs.Om0, inputs.w0, inputs.rate_II_r, inputs.logL_snIa, inputs.sigma_snIa, \
            inputs.logL_snII,inputs.sigma_snII,inputs.Z], lnL_init))
        p0.append(numpy.random.randn(ndim)*sigs + mns )
    # p0 = [numpy.random.randn(ndim)*sigs + mns for i in range(nwalkers)]

    dco = 1e-11 #measurement error very small



    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[observation['counts'],
        observation['specz'], numpy.zeros(nTrans)+dco, observation['zprob'], observation['spectype']], pool=pool)
    sampler.run_mcmc(p0, 2000)
    pool.close()

    output = open('data.pkl', 'wb')
    pickle.dump(sampler.chain, output)
    output.close()


def results():
    pkl_file = open('data.pkl', 'rb')
    data = pickle.load(pkl_file)
    samples = data[:, 100:, :].reshape((-1, data.shape[2]))
    pkl_file.close()
    fig = corner.corner(samples[:,:8], labels=["$\Omega_M$", "$w_0$", "$r_{snII}$", "$\ln{L_{snIa}}$","$\sigma_{snIa}$", "$\log{L_{snII}}$", "$\sigma_{snII}$", "$Z$"])
                      #truths=[m_true, b_true, np.log(f_true)])
    fig.savefig("triangle.png")

runModel()
#results()
#pgm()
