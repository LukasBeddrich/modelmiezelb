"""
###############################################################################
############################### OUTDATED MODULE ###############################
######################### FOR REFERENCE PRUPOSES ONLY #########################
###############################################################################

This module contains the functionality to analyze MIEZE
S(q, tau) data including
- explicit MIEZE-phase calculation
- spectral shape function / scattering function in accordance with
  asymptotic renormalization group theory for isotropic
  Heisenberg ferromagnets.

Possibly a faster implementation using cython is necessary.
"""
### IMPORTS
import numpy as np
from .utils.util import *
from scipy.integrate import quad

k_B = k_B_meV_per_K

################################################################################
### IMPLEMENTATION OF ASY. RENORM. GROUP THEORY FUNCTIONS
#continius variable energy

def gaussian(x, mu, A, e0width, kappa): # sig):
    return 1/np.sqrt(2*np.pi*e0width**2)*np.exp(-np.power(x - mu, 2.) / (2 * np.power(e0width, 2.)))

def lorentzian( x, x0, A, e0width, kappa):  ##, gam ):
    return 1/(e0width*np.pi)*e0width**2 / ( e0width**2 + ( x - x0 )**2)

def fqe_I(q, energy, A, energy_c, kappa, **kwargs):
    """
    Analytical expression for the dynamical correlation function
    of an isotropic ferromagnet in first order in
    epsilon = 6 - d dimensions introduced by Iro.

    Parameter
    ---------
    q               :   float
        momentum transfer [1/angstroem]
    energy          :   float
        energy transfer [meV]
    A               :   float
        proportionality factor for linewidth calculation in gamma = A * q**2.5
        [A] = meV angstroem**(5/2) | A(nickel) = 350 meV angstroem**(5/2)
    energy_c        :   float
        analogon to the linewidth in a Lorentzian spectral shape function
        Has to given in the same unit as energy.
    kappa           :   float
        inverse (mag.) correlation length [angstroem]

    Return
    ------
    fqe             :   float
        spectral shape function F(q, energy)

    """
    return 1 / np.pi / energy_c * np.real(1.0 / (-1j * (energy/energy_c) + 1/(capital_z(q/kappa) * capital_pi1(q/kappa, capital_w(q, energy, A)))))

def capital_z(x):
    """
    Calculates Z value for fqe_I

    Parameters
    ----------
    x               :   float
        q (momentum transfer) / kappa (inverse mag. correlation length)

    Return
    ------
    Z               :   float
    """
    a = 0.46
    b = 3.16
    k = 0.51
    return 1.0/(1 - k * np.arctan(a * (1 + 1/x**2)/(1 + b/x**2)**2)) * (1 + b/x**2)**(-3./4)

def capital_pi1(x, w):
    """
    Calculates PI_1 value for fqe_I
    Related to the self energy of the dynamic susceptibility

    Parameters
    ----------
    x               :   float
        q (momentum transfer) / kappa (inverse mag. correlation length)
    w               :   float

    Return
    ------
    PI_1            :   float
    """
    a = 0.46
    b = 3.16
    return ((1 + b/x**2)**(2 - 3.0/4) - 1j * a * w)**(3.0/5)

def capital_w(q, energy, A):
    """
    Calculates W value for PI_1 in fqe_I

    Parameters
    ----------
    q               :   float
        momentum transfer [1/angstroem]
    energy          :   float
        energy transfer [meV]
    A               :   float
        proportionality factor for linewidth calculation in gamma = A * q**2.5
        [A] = meV angstroem**(5/2) | A(nickel) = 350 meV angstroem**(5/2)

    Return
    ------
    W               :   float
    """

    return energy * capital_z(np.inf) / A / q**2.5

################################################################################
### MIEZE SECTION

#fqe_I(q, energy, A, energy_c, kappa, **kwargs)
"""
Scattering function (line 219 - 258)

This is what is ultimately measured by most (inelastic) neutron scattering techniques
In the code it uses the lineshape functions and shifts it according to the energy transfer to +e0 and -e0
write it so that in general an arbitrary amount of lineshapes at different e0 values can occure (always in pairs +e0 and -e0)
"""

def SvqE(func,e, e0, e0width, bckg, amplitude, lam, T, q, kappa, A):  ##plot
    """
    Calculates scattering function for spin-waves in a Heisenberg ferromagnet
    assuming a spectral shape function derived within the asymptotic
    renormalization group theory in epsilon = 6 - d dimensions.

    Parameters
    ----------
    func        : function
        gaussian lorentzian or fqe_I
    e0          :   float/array
        excitation energy [meV]
    e0          :   float
        excitation energy [meV]
    e0width     :   float
        width of the excitation [meV]
    bckg        :   float
        background in the energy domain between [0,1)
    amplitude   :   float
        adjustable amplitude of the scattering function
    lam         :   float
        wavelength of the neutrons [angstroem]
    T           :   float
        temperature of the sample [K]
    q           :   float
        (reduced) momentum transfer [1/angstroem]
    kappa       :   float
        inverse correlation length [1/angstroem]
    A           :   float
        proportionality factor in excitation linewidth equation
        Gamma = A * q**2.5

    Return
    ------
    sqE         :   float
        Scattering function value
    """
    
    Ng0s,Nl0s,Sum=0,0,0
    N=2  #### ASK LUKAS
    Ng0= (1 - bckg )
    if (np.size(e0) ==1):
        e0=np.array([e0])

    Ng0s = Ng0*np.size(e0)

    for e0s in e0:

        Nl0s += Ng0*(np.exp(-e0s/(k_B*T)))
        Sum  += Ng0*(func(q, e+e0s, A, e0width, kappa)+(np.exp(-e0s/(k_B*T)))*func(q, e-e0s, A, e0width, kappa))

    return amplitude/(Ng0s+Nl0s+N*bckg)*(Sum+N*bckg * constant_normed(-1 * energy_from_lambda(lam), 15.0 - energy_from_lambda(lam)))


def SvqE_Eint_lamInt(func,freq, e0, e0width, n_e, n_lam, lsd, lam_c, lam_w, time, bckg, amplitude, T, q, kappa, A):
    """
    Calculates the S(q,tau) intermediate scattering function

    Parameters
    ----------
    func        : function
        gaussian lorentzian or fqe_I
    e0          :   float/array
        excitation energy [meV]
    e0width     :   float
        width of the excitation [meV]
    n_e         :   int
        number of points for energy integration
        2000 < n_e < 20000
    n_lam       :   int
        number of points for wavelength integration
        20 < n_e < 200
    lsd         :   float
        sample-detector-distance [m]
    lam_c       :   float
        center wavelength of the triangular wavelength distribution
        from a velocity selector
    lam_w       :   float
        FWHM of the velocity selectors wavelength distribution
    time        :   int
        1: adds pi/2 to the MIEZE phase
        0: no shift
    bckg        :   float
        background in the energy domain between [0,1)
    amplitude   :   float
        adjustable amplitude of the scattering function
    lam         :   float
        !!! This variable is not referenced anywhere in the function!!! DELETA?
        wavelength of the neutrons [angstroem]
    T           :   float
        temperature of the sample [K]
    q           :   float
        (reduced) momentum transfer [1/angstroem]
    kappa       :   float
        inverse correlation length [1/angstroem]
    A           :   float
        proportionality factor in excitation linewidth equation
        Gamma = A * q**2.5

    Return
    ------
    sqt  :   float
        Intermediate scattering function using explicit calc of the MIEZE phase
    """


    l = np.linspace(1-lam_w*1.01, 1+lam_w*1.01, n_lam) * lam_c
    ll = np.tile(l,(n_e,1))
    a = -0.99999 * energy_from_lambda(l)
    b = a + 15.0
    ee = np.linspace(a, b, n_e)
    phi_det = time*np.pi/2

    sqe = SvqE(func,ee, e0, e0width, bckg, amplitude, ll, T, q, kappa, A)
    phase_weight = np.cos(MIEZE_phase(ee, freq, lsd, ll) + phi_det)
    det_eff = DetFac(ee, ll, 1) # detector efficiency correction on
    tri_distr = triangle_distribution(ll, lam_c, lam_w)
    return np.trapz(np.trapz(det_eff * sqe * phase_weight * tri_distr, ee, axis=0), l)
#The transforms of the scattering function S(q,omega) into the intermediate scattering function S(q,tau) explanation of what it does.

##################################################
######## Detector efficiency correction ##########
##################################################
def DetFac(e, lam, on):
    return (6*(0.0233+0.079*(0.5*np.log10(81.82) - 0.5*np.log10(81.92/lam**2 + e)))-1)*on+1

#corrections (line 326-454)
def DetFac_Eint_lamInt(func,e0, e0width, n_e, n_lam, lam_c, lam_w, on, bckg, amplitude, T, q, kappa, A):
    """
    Calculates the correction factor for the wavelength dependent
    CASCADE detector efficiency.
    det_eff = DetFac_Eint_lamIntpy(..., on = 0) / DetFac_Eint_lamIntpy(..., on = 1)

    Parameters
    ----------
    e0          :   float
        excitation energy [meV]
    e0width     :   float
        width of the excitation [meV]
    n_e         :   int
        number of points for energy integration
        2000 < n_e < 20000
    n_lam       :   int
        number of points for wavelength integration
        20 < n_e < 200
    lam_c       :   float
        center wavelength of the triangular wavelength distribution
        from a velocity selector
    lam_w       :   float
        FWHM of the velocity selectors wavelength distribution
    on          :   int
        1: activates the weighted integration over S(q,E) * det_eff(lam)
        0: deactivates it
    bckg        :   float
        background in the energy domain between [0,1)
    amplitude   :   float
        adjustable amplitude of the scattering function
    lam         :   float
        wavelength of the neutrons [angstroem]
    T           :   float
        temperature of the sample [K]
    q           :   float
        (reduced) momentum transfer [1/angstroem]
    kappa       :   float
        inverse correlation length [1/angstroem]
    A           :   float
        proportionality factor in excitation linewidth equation
        Gamma = A * q**2.5

    Return
    ------
    det_eff0/1  :   float
        Divisor / Dividend to calculate detector efficiency correction
    """
    l = np.linspace(1-lam_w*1.01, 1+lam_w*1.01, n_lam) * lam_c
    ll = np.tile(l,(n_e,1))
    a = -0.99999 * energy_from_lambda(l)
    b = 15.0 + a
    ee = np.linspace(a, b, n_e)
    det_eff = DetFac(ee, ll, on)
    sqe = SvqE(func,ee, e0, e0width, bckg, amplitude, ll, T, q, kappa, A)
    tri_distr = triangle_distribution(ll, lam_c, lam_w)
    return np.trapz(np.trapz(det_eff * sqe * tri_distr, ee, axis=0), l)

##################################################
######## Energy cut-off correction ###############
##################################################
# since there is a finite incoming wavelength, the energy integral is cut-off at the energy loss side. This leads to improper scaling of the Lorenzians.
# therefore, a rescaling factor needs to be calculated numerically by comparing an ideal integral and the cut-off integral
# final function described in more detail

def CutFac_Eint(func,e0, e0width, n_e, n_lam, lam_c, lam_w, on, bckg, amplitude, T, q, kappa, A):
    """
    Calculates the correction factor for the S(q,t) curve
    which arises from the integration cut-off in energy-space
    due to the limited amount of energy a neutron can transfer
    to the system.
    usage: A_energy_cutoff = CutFac_Eintpy(..., on = 0) / CutFac_Eintpy(..., on = 1)

    Parameters
    ----------
    func        : function
        gaussian lorentzian or fqe_I
    e0          :   float/array
        excitation energy [meV]
    e0width     :   float
        width of the excitation [meV]
    n_e         :   int
        number of points for trapezoid integration algorithm
    on          :   float/int
        activates cut-off integration (on=1)
        or full integration (on=0)
    bckg        :   float
        background in the energy domain between [0,1)
    amplitude   :   float
        adjustable amplitude of the scattering function
    lam         :   float
        wavelength of the neutrons [angstroem]
    T           :   float
        temperature of the sample [K]
    q           :   float
        (reduced) momentum transfer [1/angstroem]
    kappa       :   float
        inverse correlation length [1/angstroem]
    A           :   float
        proportionality factor in excitation linewidth equation
        Gamma = A * q**2.5

    Return
    ------
                :   numpy.ndarray (1D)
        Energy integrated scattering function with (on=1) or without (on=0)
        wavelength dependent detector efficiency.

    NOTE
    ----
    This returns a 1D array since it depends on the wavelength/starting point of energy integration
    """
    # if usage is CutFac_Eintpy(..., on = 0) then
    # assigned variables usually are the last ones in the fucntion definition, so we can avoid to write them easily
    # def CutFac_Eint( ..., on=0) making the 0/1 as predefined value, instead of def CutFac_Eint(..., on, ...)
    # also control statemenst will be good to check if on is bolean or not, and functions with more than 5
    # arguments are not recommended by pep8 / style guides.

    '''
    if (func=="gaussian"):
        print("Gaussian lineshape used")
        func=gaussian
    elif(func=="lorentzian"):
        print("Lorentzian lineshape used")
        func=lorentzian
    elif(func=="fqe_I"):
        print("Fqe_I lineshape used")
        func=fqe_I
    elif(hasattr(func, '__call__')):
        print("Using user defined function")
    else:
        print("Introduce user defined function or use predefined gaussian lorentzian fqe_I as a str")
    '''


    l = np.linspace(1-lam_w*1.01, 1+lam_w*1.01, n_lam) * lam_c
    ll = np.tile(l,(n_e,1))
    a = -0.99999 * energy_from_lambda(l) ## a is defined also inside if and not used in else, so here is useless.
    #b = 15.0 + a                         ##This b is not used.
    if on:
        es = np.linspace(a, 15.0 + a, n_e)
        return np.trapz(SvqE(func,es, e0, e0width, bckg, amplitude, ll, T, q, kappa, A), es, axis=0)
    es = np.tile(np.linspace(-15.0, 15.0, 2 * n_e), (n_lam, 1)).T
    return np.trapz(SvqE(func,es, e0, e0width, bckg, amplitude, np.tile(l,(2*n_e, 1)), T, q, kappa, A), es, axis=0)

##################################################
######## Intermediate scattering function ########
##################################################

def Sqt(func, freq, e0, e0width, n_e, n_lam, l_SD, lam_c, lam_w, bckg, amplitude, T, q, kappa, A):
    """

    """
    phys_Int = np.array(CutFac_Eint(func,e0, e0width, n_e, n_lam, lam_c, lam_w, 1, bckg, amplitude, T, q, kappa, A))
    real_Int = np.array(CutFac_Eint(func,e0, e0width, n_e, n_lam, lam_c, lam_w, 0, bckg, amplitude, T, q, kappa, A))
    A1 = np.mean(real_Int/phys_Int)

    fac_on = np.array(DetFac_Eint_lamInt(func,e0, e0width, n_e, n_lam, lam_c, lam_w, 1, bckg, amplitude, T, q, kappa, A))
    fac_off  = np.array(DetFac_Eint_lamInt(func,e0, e0width, n_e, n_lam, lam_c, lam_w, 0, bckg, amplitude, T, q, kappa, A))
    A2   = fac_off/fac_on

    func0 = [SvqE_Eint_lamInt(func,f, e0, e0width, n_e, n_lam, l_SD, lam_c, lam_w, 0, bckg, amplitude, T, q, kappa, A) for f in freq]
    func1 = [SvqE_Eint_lamInt(func,f, e0, e0width, n_e, n_lam, l_SD, lam_c, lam_w, 1, bckg, amplitude, T, q, kappa, A) for f in freq]
    return A1*A2*np.sqrt(np.array(func0)**2+np.array(func1)**2)
#    return A2*np.sqrt(np.array(func0)**2+np.array(func1)**2)

################################################################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt