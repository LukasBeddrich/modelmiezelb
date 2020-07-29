"""
Mathematical functions for line shapes in S(q,E) modelling.
"""

import numpy as np

###############################################################################
###############################################################################
###############################################################################

def lorentzian(x, x0, width, **kwargs):
    """
    Calculates a Lorentzian / Cauchy probability distribution.
    Normalized to 1.

    Parameters
    ----------
    x : float, ndarray
        running variable in our context mostly energy
    x0 : float
        mean value of the distribution
    width : float
        FWHM of the distributopm
    
    Returns
    -------
        : float, ndarray
        probability distribution function results

    """
    return width / np.pi / ((x - x0)**2 + width**2)

#------------------------------------------------------------------------------

def lorentzian_cdf(x, x0, width, **kwargs):
    """
    Calculates the cumulative distribution function of a Lorentzian / 
    Cauchyy distribution

    Parameters
    ----------
    x : float, ndarray
        running variable in our context mostly energy
    x0 : float
        mean value of the distribution
    width : float
        FWHM of the distributopm
    
    Returns
    -------
        : float, ndarray
        integrated value of the probability distribution function
    """
    return 1.0/np.pi * np.arctan((x-x0)/width) + 0.5

#------------------------------------------------------------------------------

def gaussian(x, x0, sigma):
    """
    Calculates a Gaussian / normal probability distribution.
    Normalized to 1.

    Parameters
    ----------
    x : float, ndarray
        running variable in our context mostly energy
    x0 : float
        mean value of the distribution
    sigma : float
        standard deviation of the distributopm
    
    Returns
    -------
        : float, ndarray
        probability distribution function results

    """
    return np.exp(-0.5 * np.power((x - x0)/sigma, 2)) / np.sqrt(2*np.pi) / np.abs(sigma)

#------------------------------------------------------------------------------

def fqe_I(e, e_c, A, q, kappa):
    """
    Analytical expression for the dynamical correlation function
    of an isotropic ferromagnet in first order in
    epsilon = 6 - d dimensions introduced by Iro.

    Parameter
    ---------
    e               :   float, ndarray
        running variable, energy transfer [meV]
    e_c        :   float
        analogon to the linewidth in a Lorentzian spectral shape function
        Has to given in the same unit as energy variable 'e'.
    A               :   float
        proportionality factor for linewidth calculation in gamma = A * q**2.5
        [A] = meV angstroem**(5/2) | A(nickel) = 350 meV angstroem**(5/2)
    q               :   float
        momentum transfer [1/angstroem]
    kappa           :   float
        inverse (mag.) correlation length [angstroem]

    Return
    ------
    fqe             :   float
        spectral shape function F(q, energy)

    """
    return 1 / np.pi / e_c * np.real(1.0 / (-1j * (e/e_c) + 1/(capital_z(q/kappa) * capital_pi1(q/kappa, capital_w(q, e, A)))))

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