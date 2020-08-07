"""

"""
import json
###############################################################################
from numpy import trapz, arange, ones
from scipy.integrate import quad
from .sqe_model import SqE, UPPER_INTEGRATION_LIMIT
from .utils.util import detector_efficiency, triangle_distribution

###############################################################################

class CorrectionFactory:
    _registry = {}

    @classmethod
    def register(cls, specifier):
        def wrapper(wrapped_class):
            if specifier in cls._registry:
                raise KeyError(f"'{specifier}' is already in factory's registry.")
            cls._registry[specifier] = wrapped_class
            return wrapped_class
        return wrapper

#------------------------------------------------------------------------------

    @classmethod
    def create(cls, specifier, **kwargs):
        if specifier not in cls._registry:
            raise KeyError(f"'{specifier}' is not in factory's registry.")
        creator = cls._registry[specifier]
        return creator

#------------------------------------------------------------------------------

    @classmethod
    def manual_register(cls, specifier, correction):
        assert isinstance(correction, CorrectionFactor)
        if specifier in cls._registry:
            raise KeyError(f"'{specifier}' is already in factory's registry.")
        cls._registry[specifier] = correction

###############################################################################
###############################################################################
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

    @classmethod
    def load_from_dict(cls, **correction_dict):
        """

        """
        sqe = SqE.load_from_dict(**correction_dict.pop('sqe'))
        specifier = correction_dict.pop('specifier')
        return CorrectionFactory.create(specifier)(sqe, **correction_dict["calc_params"])

#------------------------------------------------------------------------------

    @classmethod
    def load_from_jsonfile(cls, json_file):
        """

        """
        with open(json_file, "r") as jsonfile:
            imported_correction_dict = json.load(jsonfile)

        specifier = imported_correction_dict["specifier"]
        for v in imported_correction_dict["sqe"].values():
            if 'domain' in v.keys(): v['domain'] = tuple(v['domain'])

        return CorrectionFactory.create(specifier).load_from_dict(**imported_correction_dict)

#------------------------------------------------------------------------------

    def export_to_dict(self):
        """

        """
        raise NotImplementedError

#------------------------------------------------------------------------------

    def export_to_jsonfile(self, json_file):
        """
        Saves a JSON-file, which can be used to create
        a new instance of the same class via the 
        cls.load_from_jsonfile method as alternate constructor.

        Parameters
        ----------
        json_file   :   str
            path specifying the JSON file for saving the Line object.

        Returns
        -------
            :   NoneType
        """
        with open(json_file, "w") as jsonfile:
            json.dump(self.export_to_dict(), jsonfile, indent=4)
        return None

#------------------------------------------------------------------------------

    def calc(self, var):
        """

        """
        raise NotImplementedError

###############################################################################
###############################################################################
###############################################################################

@CorrectionFactory.register('detector_eff')
class DetectorEfficiencyCorrectionFactor(CorrectionFactor):
    """

    """

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

#------------------------------------------------------------------------------

    def export_to_dict(self):
        """
        Returns a dictionary, which can be used to create
        a new instance of the same class via the 
        cls.load_from_dict method as alternate constructor.

        Returns
        -------
            :   dict
            dictionary of the contents of the Transformer object.
        """
        export_dict = dict(calc_params=self.calc_params, specifier='detector_eff')
        export_dict['sqe'] = self.sqe.export_to_dict()
        return export_dict

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

@CorrectionFactory.register('energy_cutoff')
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

    def export_to_dict(self):
        """
        Returns a dictionary, which can be used to create
        a new instance of the same class via the 
        cls.load_from_dict method as alternate constructor.

        Returns
        -------
            :   dict
            dictionary of the contents of the Transformer object.
        """
        export_dict = dict(calc_params=self.calc_params, specifier='energy_cutoff')
        export_dict['sqe'] = self.sqe.export_to_dict()
        return export_dict

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