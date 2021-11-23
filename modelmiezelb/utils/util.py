import numpy as np
from scipy.integrate import quad

### PHYSICAL CONSTANTS
m_n     = 1.674927471e-27 # kg
h_eV    = 4.135667662e-15 # eV*s
h_J     = 6.626070040e-34 # J*s
hbar_eV = h_eV/(2*np.pi)
hbar_J  = h_J/(2*np.pi)
eV      = 1.602176565e-19 # J
k_B_meV_per_K = 8.617333262e-02 # meV/K

#------------------------------------------------------------------------------

### CASCADE DETECTOR PARAMETERS
na = 6.022*10**23 # Avogadro constant
rho = 2.34
a = 10.81
sigma = 3837.
ddrift = 1.
dgem = 1.1
eta = 90.
iso = 100.

#------------------------------------------------------------------------------

### Utilities
def constant_normed(start, stop):
	return 1.0 / (stop - start)

#------------------------------------------------------------------------------
### COMPREHENSIVE DESCRIPTION OF THE CASCADE DETECTOR EFFICIENCY

def mu(lam, iso):
    return rho * na/a * sigma*iso/100*10**(-24) / 10000*lam/1.8

def Pvor(x, d, r, alpha):
    return (1 - (d - x*np.cos(alpha))/r)/2

def Prueck(x, r, alpha):
    return (1 - x*np.cos(alpha)/r)/2

def integer1(x, lam, iso, d, r, alpha):
    return mu(lam, iso)*np.exp(-mu(lam,  iso)*x)*Pvor(x, d, r, alpha)

def integer2(x, lam, iso, r, alpha):
    return mu(lam, iso)*np.exp(-mu(lam,  iso)*x)*Prueck(x, r, alpha)

def B10SchraegVor(d, u, o, r, lam, alpha, iso):
    return quad(integer1, u, o, args=(lam, iso, d, r, alpha))[0]

def B10SchraegRueck(u, o, r, lam, alpha, iso):
    return quad(integer2, u, o, args=(lam, iso, r, alpha))[0]

def B10EinfachSchraegVor(d, r, lam, alpha, iso):
    if d > r:
        return B10SchraegVor(d, (d - r)/np.cos(alpha), d/np.cos(alpha), r, lam, alpha, iso)
    return B10SchraegVor(d, 0., d/np.cos(alpha), r, lam, alpha, iso)

def B10EinfachSchraegRueck(d, r, lam, alpha, iso):
    if d > r:
        return B10SchraegRueck(0., r/np.cos(alpha), r, lam, alpha, iso)
    return B10SchraegRueck(0., d/np.cos(alpha), r, lam, alpha, iso)

def B10EffSchraegVor(d, lam, eta, iso):
    return 0.94*(B10EinfachSchraegVor(d, 3.16, lam, 90. - eta, iso) + B10EinfachSchraegVor(d, 1.53, lam, 90. - eta, iso)) + \
        0.06*(B10EinfachSchraegVor(d, 3.92, lam, 90. - eta, iso) + B10EinfachSchraegVor(d, 1.73, lam, 90. - eta, iso))

def B10EffSchraegRueck(d, lam, eta, iso):
    return 0.94*(B10EinfachSchraegRueck(d, 3.16, lam, 90. - eta, iso) + B10EinfachSchraegRueck(d, 1.53, lam, 90. - eta, iso)) + \
        0.06*(B10EinfachSchraegRueck(d, 3.92, lam, 90. - eta, iso) + B10EinfachSchraegRueck(d, 1.73, lam, 90. - eta, iso))

def RestIB10(d, lam, eta, iso):
    return np.exp(-mu(lam, iso)*d/np.cos(90-eta))

def Mieze6(lam):
    return B10EffSchraegVor(ddrift, lam, eta, iso) + \
            RestIB10(ddrift, lam, eta, iso) * B10EffSchraegRueck(dgem, lam, eta, iso) + \
            RestIB10(ddrift + dgem, lam, eta, iso) * B10EffSchraegRueck(dgem, lam, eta, iso) + \
            RestIB10(ddrift + 2*dgem, lam, eta, iso) * B10EffSchraegVor(dgem, lam, eta, iso) + \
            RestIB10(ddrift + 3*dgem, lam, eta, iso) * B10EffSchraegVor(dgem, lam, eta, iso) + \
            RestIB10(ddrift + 4*dgem, lam, eta, iso) * B10EffSchraegRueck(2*ddrift, lam, eta, iso)

def Mieze10(lam):
    return B10EffSchraegVor(ddrift, lam, eta, iso) + \
            RestIB10(ddrift, lam, eta, iso) * B10EffSchraegRueck(dgem, lam, eta, iso) + \
            RestIB10(ddrift + dgem, lam, eta, iso) * B10EffSchraegRueck(dgem, lam, eta, iso) + \
            RestIB10(ddrift + 2*dgem, lam, eta, iso) * B10EffSchraegRueck(dgem, lam, eta, iso) + \
            RestIB10(ddrift + 3*dgem, lam, eta, iso) * B10EffSchraegRueck(dgem, lam, eta, iso) + \
            RestIB10(ddrift + 4*dgem, lam, eta, iso) * B10EffSchraegVor(dgem, lam, eta, iso) + \
            RestIB10(ddrift + 5*dgem, lam, eta, iso) * B10EffSchraegVor(dgem, lam, eta, iso) + \
            RestIB10(ddrift + 6*dgem, lam, eta, iso) * B10EffSchraegVor(dgem, lam, eta, iso) + \
            RestIB10(ddrift + 7*dgem, lam, eta, iso) * B10EffSchraegVor(dgem, lam, eta, iso) + \
            RestIB10(ddrift + 8*dgem, lam, eta, iso) * B10EffSchraegRueck(2*ddrift, lam, eta, iso)

DetFacpy1 = np.vectorize(Mieze6)
DetFacpy2 = np.vectorize(Mieze10)

def cascade_efficiency(de, lam0, case=10):
    """
    Calculates an approximation for the CASCADE detector detection efficiency.
    Two cases for:
    1. 10 detector foils
    2.  6 detector foils

    Parameters
    ----------
    de      :   float
        (kinetic) energy CHANGE of the neutron in meV
    lam0    :   float
        initial wavelength in angstroem
    case    :   int
        1. 10
        2.  6

    Returns
    -------
    deteff  :   float
        CASCADE detection efficiency

    NOTE
    ----
    THIS IS FUCKED UP SINCE MODELMIEZELB USES dE = Ef - Ei ...
    """
    lamf = from_energy_to_lambdaf(de, lam0)
    if case == 10:
        return DetFacpy2(lamf)
    elif case == 6:
        return DetFacpy1(lamf)
    else:
        raise ValueError(f"You set case = {case}. Only 10 and 6 are valid arguments.")

#------------------------------------------------------------------------------

def energy_lambda_symetric_nrange(elow, eup, lam, dlam, ne, nlam):
    """
    Spans energy and lambda arrays for calculations

    Parameters
    ----------
    elow    :   float
        lower energy boundary
    eup     :   float
        upper energy boundary
    lam     :   float
        center wavelength
    dlam    :   float
        relative one sided width of the triangular wavelength distr.
    ne     :   int
        number of energy points
    nlam   :   int
        number of lambda points

    Returns
    -------
    ee      :   numpy.ndarray
        dim. (ne, nlam) energy array
    ll      :   numpy.ndarray
        dim. (ne, nlam) wavelength array
    """

    l = lam * np.linspace(1-dlam, 1+dlam, nlam)
    e = np.linspace(elow, eup, ne)
    return np.meshgrid(l, e)

#------------------------------------------------------------------------------

def energy_lambda_nrange(eup, lam, dlam, ne, nlam):
    """
    Spans energy and lambda arrays for calculations, wehere lower
    energy boudary is given by neutron wavelength

    Parameters
    ----------
    eup     :   float
        upper energy boundary
    lam     :   float
        center wavelength
    dlam    :   float
        relative one sided width of the triangular wavelength distr.
    ne     :   int
        number of energy points
    nlam   :   int
        number of lambda points

    Returns
    -------
    ee      :   numpy.ndarray
        dim. (ne, nlam) energy array
    ll      :   numpy.ndarray
        dim. (ne, nlam) wavelength array
    """

    l = lam * np.linspace(1-dlam, 1+dlam, nlam)
    ll = np.tile(l,(ne,1))
    a = -0.99999 * energy_from_lambda(l)
    ee = np.linspace(a, eup, ne)
    assert ee.shape == ll.shape
    return ee, ll

#------------------------------------------------------------------------------

def energy_lambda_symetric_drange(elow, eup, lam, dlam, step_e, step_lam):
    """
    Spans energy and lambda arrays for calculations

    Parameters
    ----------
    elow        :   float
        lower energy boundary
    eup         :   float
        upper energy boundary
    lam         :   float
        center wavelength
    dlam        :   float
        relative one sided width of the triangular wavelength distr.
    step_e      :   float
        step between energy points
    step_lam    :   int
        step between lambda points

    Returns
    -------
    ee      :   numpy.ndarray
        dim. (ne, nlam) energy array
    ll      :   numpy.ndarray
        dim. (ne, nlam) wavelength array
    """

    return energy_lambda_symetric_nrange(
        elow,
        eup,
        lam,
        dlam,
        ne=int((eup - elow)/step_e) + 1,
        nlam=int(2*dlam * lam / step_lam) + 1
    )

    # l = lam * np.arange(1-dlam, 1+dlam+(step_lam/lam/2), step_lam/lam)
    # e = np.linspace(elow, eup + step_e/2, step_e)
    # return np.meshgrid(l, e)

#------------------------------------------------------------------------------

def energy_lambda_drange(eup, lam, dlam, step_e, step_lam):
    """
    Spans energy and lambda arrays for calculations, where lower
    energy boundary is given by neutron wavelength

    Parameters
    ----------
    eup         :   float
        upper energy boundary
    lam         :   float
        center wavelength
    dlam        :   float
        relative one sided width of the triangular wavelength distr.
    step_e      :   float
        step between energy points
    step_lam    :   int
        step between lambda points

    Returns
    -------
    ee      :   numpy.ndarray
        dim. (ne, nlam) energy array
    ll      :   numpy.ndarray
        dim. (ne, nlam) wavelength array
    """

    return energy_lambda_nrange(
        eup,
        lam,
        dlam,
        ne=int((eup - -0.99999 * energy_from_lambda(lam))/step_e) + 1,
        nlam=int(2*dlam * lam / step_lam) + 1
    )

#------------------------------------------------------------------------------

### CONVERSION FUNCTIONS
def energy_from_lambda(lam):
    """Takes lambda in angstroem and gives energy in meV"""
    if hasattr(lam, "__len__"):
        return np.array([h_J**2/(2*m_n*(l*1e-10)**2)/eV*1e3 for l in lam])
    return h_J**2/(2*m_n*(lam*1e-10)**2)/eV*1e3

#------------------------------------------------------------------------------

def from_energy_to_lambdaf(de, lam0):
    """
    Calculates the wavelength of a neutron after transfering energy de and
    initial wavelength lami.

    Parameters
    ----------
    de      :   float
        (kinetic) energy CHANGE of the neutron in meV
    lam0    :   float
        initial wavelength in angstroem
    
    Returns
    -------
    lamf    :   float
        final wavelength in angstroem

    NOTE
    ----
    THIS IS FUCKED UP SINCE MODELMIEZELB USES dE = Ef - Ei ...
    """

    ei = energy_from_lambda(lam0)
    ef = ei + de # THIS IS FUCKED UP SINCE MODELMIEZELB USES dE = Ef - Ei ...
    return wavelength_from_energy(ef)

#------------------------------------------------------------------------------

def wavelength_from_energy(energy):
    """
    Calculates the wavelength of a neutron from its energy

    Parameters
    ----------
    energy      :   float, ndarray
        neutron energy in meV
    
    Return
    ------
    wavelength  :   float, ndarray
        wavelength of neutron in angstroem
    """
    return 9.045 / np.sqrt(energy)

#------------------------------------------------------------------------------

def detector_efficiency(energy, lam, on):
    """
    Efficiency of the CASCADE detector depending on the initial wavelength and
    energy transfer.

    Parameters
    ----------
    energy  :   float, ndarray
        energy transfer
    lam     :   float, ndarray
        initial wavelength of the neutron
    on      :   bool, int
        True, 1 for efficiency included
        False, 0 for efficiency neglected
    
    Returns
    -------
            :   float, ndarray
        efficiency factor of the cascade detector
    """
    return (6*(0.0233+0.079*(0.5*np.log10(81.82) - 0.5*np.log10(81.92/lam**2 + energy)))-1) * on + 1

#------------------------------------------------------------------------------

def velocity_from_lambda(lam):
    """Takes lambda in angstroem and gives the velocity in m/s"""
    return h_J/(m_n*lam*1e-10)

#------------------------------------------------------------------------------

def MIEZE_DeltaFreq_from_time(miezetime, lsd, lam):
    """
    Calculates the MIEZE frequency (omega_B-omega_A) for operation with pi-flip in Hz.
    Takes miezetime in s, sample detector distance lsd in m and the wavelength lam in AA
    """
    return miezetime*m_n*velocity_from_lambda(lam)**3/(4*np.pi*hbar_J*lsd)

#------------------------------------------------------------------------------

### explicit MIEZE phase calculation
def MIEZE_phase(energy, freq, lsd, lam):
    """
    Explicit MIEZE phase applying a pi-flip
    energy - in meV, frequency freq - in Hz, sample detector distance lsd - in m, wavelength lam - in AA'
    """
    vel = velocity_from_lambda(lam)
    return 4.0*np.pi*lsd*freq*(1/vel-1/(np.sqrt(vel**2+2/m_n*energy*eV*1e-3)))

#------------------------------------------------------------------------------

### TRIANGULAR WAVELENGTH DISTRIBUTION OF A VELOCITY SELECTOR
def triangle_distribution(x, m, FWHM):
    """
    Triangular wavelength distribution.
    Takes lambda x, center lambda m, and FWHM lam_width
    """
    l = m-m*FWHM
    r = m+m*FWHM
#    if l<=x and x<=m:
#        return ((x-l)/(m-l))/(m-l)
#    elif m<=x and x<=r:
#        return (1-(x-m)/(r-m))/(m-l)
#    else:
#        return 0

    left_side = np.where(np.logical_and(l <= x, x <= m), (x-l)/(m-l)/(m-l), 0.0)
    right_side = np.where(np.logical_and(m <= x, x <= r), (1-(x-m)/(r-m))/(m-l), 0.0)
    return left_side + right_side

#------------------------------------------------------------------------------

def bose_factor(e, T):
    """
    Parameters
    ----------
    e   :   float
        Energy value of the excitation [meV]
    T   :   float
        Temperture of the sample [K]

    Returns
    -------
    n_B :   float
        Bose factor
    """
    return np.exp(-abs(e)/ k_B_meV_per_K / T)