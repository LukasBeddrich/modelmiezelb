"""

"""

from numpy import trapz, arange, ones
from scipy.integrate import quad
from .sqe_model import SqE, UPPER_INTEGRATION_LIMIT
from .utils.util import detector_efficiency, triangle_distribution

###############################################################################

class CorrectionFactor:
    """

    """

    _required_params = ("n_e", "n_lam", "on")

    def __init__(self, sqe, **calc_params):
        """

        """
        self.sqe = sqe
        self.calc_params = calc_params
#        raise NotImplementedError

#------------------------------------------------------------------------------

    def __call__(self, var):
        """

        """
        raise NotImplementedError

#------------------------------------------------------------------------------

    def calc(self, var):
        """

        """
        raise NotImplementedError

###############################################################################
###############################################################################
###############################################################################

class DetectorEfficiencyCorrectionFactor(CorrectionFactor):
    """

    """

    # def __call__(self, energy, lam):
    #     """
    #     Calculates MIEZE detector efficiency correction factor

    #     Parameters
    #     ----------
    #     energy  :   float, ndarray (n, m)
    #         energy range with energy transfer depending limited by neutron wavelength
    #     lam     :   float, ndarray (same as energy)
    #         inital wavelength of the neutrons
        
    #     Returns
    #     -------
    #             :   float, ndarray (same as input)
    #         detector efficiency correction factor
    #     """
    #     return self.calc(energy, lam, 1) / self.calc(energy, lam, 0)

    def __call__(self, energy, lam):
        """
        Calculates product of MIEZE detector efficiency factor, wavelength distr.
        and scattering function 'sqe'

        Parameters
        ----------
        energy  :   float, ndarray (n, m)
            energy range with energy transfer depending limited by neutron wavelength
        lam     :   float, ndarray (same as energy)
            inital wavelength of the neutrons
        
        Returns
        -------
                :   float, ndarray (same as input)
            detector efficiency correction factor
        """
        return self.calc(energy, lam)

#------------------------------------------------------------------------------

    def calc(self, energy, lam):
        """
        Calculates product of MIEZE detector efficiency factor, wavelength distr.
        and scattering function 'sqe'

        Parameters
        ----------
        energy  :   float, ndarray (n, m)
            energy range with energy transfer depending limited by neutron wavelength
        lam     :   float, ndarray (same as energy)
            inital wavelength of the neutrons

        Returns
        -------
                :   float, ndarray (same as input)
        """
        det_eff = detector_efficiency(energy, lam, 1)
        sqe = self.sqe(energy)
        tri_distr = triangle_distribution(
            lam,
            self.sqe.model_params["lam"],
            self.sqe.model_params["dlam"]
        )
        return trapz(trapz(sqe * tri_distr, energy, axis=0), lam[0]) / trapz(trapz(det_eff * sqe * tri_distr, energy, axis=0), lam[0])

        # det_eff_ratio = detector_efficiency(energy, lam, 0) / detector_efficiency(energy, lam, 1)
        # return det_eff_ratio


#------------------------------------------------------------------------------

    def legacy_calc(self, energy, lam, on=1):
        """
        Calculates product of MIEZE detector efficiency factor, wavelength distr.
        and scattering function 'sqe'

        Parameters
        ----------
        energy  :   float, ndarray (n, m)
            energy range with energy transfer depending limited by neutron wavelength
        lam     :   float, ndarray (same as energy)
            inital wavelength of the neutrons
        on      :   bool, int,  optional
            True, 1 for efficiency included
            False, 0 for efficiency neglected

        Returns
        -------
                :   float, ndarray (same as input)
        """
        det_eff = detector_efficiency(energy, lam, on)
        sqe = self.sqe(energy)
        tri_distr = triangle_distribution(
            lam,
            self.sqe.model_params["lam"],
            self.sqe.model_params["dlam"]
        )
        return det_eff * sqe * tri_distr

#------------------------------------------------------------------------------

    def correction(self, energy, lam, on=1):
        """
        Calculates the correction factor just as
        arg_inel_mieze_model.DetFac_Eint_lamInt

        Parameters
        ----------
        energy  :   float, ndarray (n, m)
            energy range with energy transfer depending limited by neutron wavelength
        lam     :   float, ndarray (same as energy)
            inital wavelength of the neutrons
        on      :   bool, int,  optional
            True, 1 for efficiency included
            False, 0 for efficiency neglected

        Returns
        -------
                :   float
        """
        return trapz(trapz(self.legacy_calc(energy, lam, on), energy, axis=0), lam[0])


###############################################################################
###############################################################################
###############################################################################

class EnergyCutOffCorrectionFactor(CorrectionFactor):
    """

    """

    def __call__(self, energy, lam, on=1):
        """
        Calculates MIEZE detector efficiency correction factor

        Parameters
        ----------
        energy  :   float, ndarray (n, m)
            energy range with energy transfer depending limited by neutron wavelength
        lam     :   float, ndarray (same as energy)
            inital wavelength of the neutrons
        on      :   bool, int,  optional
            True, 1 for cut off energy included
            False, 0 for cut off energy neglected
        
        Returns
        -------
                :   float, ndarray (same as input)
            array for getting the energy cut off correction factor
        """
        return self.calc(energy, lam, on)

#------------------------------------------------------------------------------

    def calc(self,energy, lam, on=1):
        """
        Calculates MIEZE detector efficiency correction factor

        Parameters
        ----------
        energy  :   float, ndarray (n, m)
            energy range with energy transfer depending limited by neutron wavelength
        lam     :   float, ndarray (same as energy)
            inital wavelength of the neutrons
        on      :   bool, int,  optional
            True, 1 for cut off energy included
            False, 0 for cut off energy neglected
        
        Returns
        -------
                :   float, ndarray (same as input)
            array for getting the energy cut off correction factor
        """

        return ones(energy.shape)

#------------------------------------------------------------------------------

    def legacy_calc(self,energy, lam, on=1):
        """
        Calculates MIEZE detector efficiency correction factor

        Parameters
        ----------
        energy  :   float, ndarray (n, m)
            energy range with energy transfer depending limited by neutron wavelength
        lam     :   float, ndarray (same as energy)
            inital wavelength of the neutrons
        on      :   bool, int,  optional
            True, 1 for cut off energy included
            False, 0 for cut off energy neglected
        
        Returns
        -------
                :   float, ndarray (same as input)
            array for getting the energy cut off correction factor
        """
        if on:
            return self.sqe(energy)
        else:
            de = (energy[-1] - energy[0]) / (len(energy) - 1)
            e = arange(-UPPER_INTEGRATION_LIMIT, UPPER_INTEGRATION_LIMIT + 0.01, de)
            return self.sqe(e)

#------------------------------------------------------------------------------

    def correction(self, energy, lam, on=1):
        """
        Calculates the correction factor just as
        arg_inel_mieze_model.CutFac_Eint

        Parameters
        ----------
        energy  :   float, ndarray (n, m)
            energy range with energy transfer depending limited by neutron wavelength
        lam     :   float, ndarray (same as energy)
            inital wavelength of the neutrons
        on      :   bool, int,  optional
            True, 1 for cut off energy included
            False, 0 for cut off energy neglected

        Returns
        -------
                :   float
        """
        if on:
            return trapz(self.legacy_calc(energy, lam, on), energy, axis=0)
        else:
            de = (energy[-1] - energy[0]) / (len(energy) - 1)
            e = arange(-UPPER_INTEGRATION_LIMIT, UPPER_INTEGRATION_LIMIT + 0.01, de)
            return trapz(self.legacy_calc(e, lam, 1), lam, axis=0)