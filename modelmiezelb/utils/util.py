import numpy as np

### PHYSICAL CONSTANTS
m_n     = 1.674927471e-27 # kg
h_eV    = 4.135667662e-15 # eV*s
h_J     = 6.626070040e-34 # J*s
hbar_eV = h_eV/(2*np.pi)
hbar_J  = h_J/(2*np.pi)
eV      = 1.602176565e-19 # J
k_B_meV_per_K = 8.617333262e-02 # meV/K

#------------------------------------------------------------------------------

### Utilities
def constant_normed(start, stop):
	return 1.0 / (stop - start)

#------------------------------------------------------------------------------

def energy_lambda_symetric_nrange(elow, eup, lam, dlam, n_e, n_lam):
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
    n_e     :   int
        number of energy points
    n_lam   :   int
        number of lambda points

    Returns
    -------
    ee      :   numpy.ndarray
        dim. (n_e, n_lam) energy array
    ll      :   numpy.ndarray
        dim. (n_e, n_lam) wavelength array
    """

    l = lam * np.linspace(1-dlam, 1+dlam, n_lam)
    e = np.linspace(elow, eup, n_e)
    return np.meshgrid(l, e)

#------------------------------------------------------------------------------

def energy_lambda_nrange(eup, lam, dlam, n_e, n_lam):
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
    n_e     :   int
        number of energy points
    n_lam   :   int
        number of lambda points

    Returns
    -------
    ee      :   numpy.ndarray
        dim. (n_e, n_lam) energy array
    ll      :   numpy.ndarray
        dim. (n_e, n_lam) wavelength array
    """

    l = lam * np.linspace(1-dlam, 1+dlam, n_lam)
    ll = np.tile(l,(n_e,1))
    a = -0.99999 * energy_from_lambda(l)
    ee = np.linspace(a, eup, n_e)
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
        dim. (n_e, n_lam) energy array
    ll      :   numpy.ndarray
        dim. (n_e, n_lam) wavelength array
    """

    return energy_lambda_symetric_nrange(
        elow,
        eup,
        lam,
        dlam,
        n_e=int((eup - elow)/step_e) + 1,
        n_lam=int(2*dlam * lam / step_lam) + 1
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
        dim. (n_e, n_lam) energy array
    ll      :   numpy.ndarray
        dim. (n_e, n_lam) wavelength array
    """

    return energy_lambda_nrange(
        eup,
        lam,
        dlam,
        n_e=int((eup - -0.99999 * energy_from_lambda(lam))/step_e) + 1,
        n_lam=int(2*dlam * lam / step_lam) + 1
    )

#------------------------------------------------------------------------------

### CONVERSION FUNCTIONS
def energy_from_lambda(lam):
    """Takes lambda in angstroem and gives energy in meV"""
    if hasattr(lam, "__len__"):
        return np.array([h_J**2/(2*m_n*(l*1e-10)**2)/eV*1e3 for l in lam])
    return h_J**2/(2*m_n*(lam*1e-10)**2)/eV*1e3

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
        Flase, 0 for efficiency neglected
    
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