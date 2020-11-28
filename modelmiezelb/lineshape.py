"""
Module for line shapes in SqE modeling.
"""

import json
import numpy as np
from types import MethodType
from math import isclose
from inspect import getfullargspec
from .utils.lineshape_functions import lorentzian, lorentzian_cdf, fqe_c, fqe_I
from .utils.util import bose_factor

###############################################################################
###############################################################################
###############################################################################

class LineFactory:
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
        return creator(**kwargs)

#------------------------------------------------------------------------------

    @classmethod
    def manual_register(cls, specifier, line):
        assert isinstance(line, Line)
        if specifier in cls._registry:
            raise KeyError(f"'{specifier}' is already in factory's registry.")
        cls._registry[specifier] = line

###############################################################################
###############################################################################
###############################################################################

class Line:
    """
    An informal interface for Line classes.
    """
    # Absolute tolerance when a line is considered quasi- instead of inelastic
    QUASI_TO_INELASTIC_DISTINCTION_VALUE = 0.0001 # meV
    INTEGRATION_WIDTH_SCALE = None
    INTEGRATION_POINT_DENSITY_PARAMETER = None
    INTEGRATION_POINT_NUMBER_FACTOR = None
    _required_params = ()

    def __init__(self, name, domain, **line_params):
        """
        Parameters
        ----------
        name        :   str
            Name or descriptor of the Line
        domain      :   tuple
            (min, max) domain on which the Line is well defined
        line_params :
            parameters to evaluate the line
        """
        assert isinstance(name, str)
        self.name = name
        assert isinstance(domain, tuple)
        self.domain = domain # range in which the Line is calculated
        assert isinstance(line_params, dict)
        self.line_params = line_params

#------------------------------------------------------------------------------
    
    def __call__(self, var):
        """
        Evaluate the line at var, if var in domain

        Parameters
        ----------
        var : float, ndarray, list, tuple
            running variable of the Line
        """
#        self.check_params()
#        var = self.within_domain(var)
        return self.calc(var, **self.line_params) * self.within_domain(var)

#------------------------------------------------------------------------------

    def get_adaptive_integration_grid(self, ne):
        """

        """
        half_grid = np.geomspace(
            1,
            self.INTEGRATION_POINT_DENSITY_PARAMETER,
            self.INTEGRATION_POINT_NUMBER_FACTOR * ne // 2
        )
        m = self.line_params["width"] * self.INTEGRATION_WIDTH_SCALE / (self.INTEGRATION_POINT_DENSITY_PARAMETER - 1)
        t = self.line_params["x0"] - m

        half_grid = m * half_grid + t
        return np.concatenate((np.flip(-half_grid) + 2 * self.line_params["x0"], half_grid[1:]))

#------------------------------------------------------------------------------

    @classmethod
    def load_from_dict(cls, **line_dict):
        """

        """
        return cls(**line_dict)

#------------------------------------------------------------------------------

    @classmethod
    def load_from_jsonfile(cls, json_file):
        """

        """
        with open(json_file, "r") as jsonfile:
            imported_line = json.load(jsonfile)
        imported_line["domain"] = tuple(imported_line["domain"])
        return cls(**imported_line)

#------------------------------------------------------------------------------

    def export_to_dict(self):
        """

        """
        pass

#------------------------------------------------------------------------------

    def export_to_jsonfile(self, json_file):
        """

        """
        pass

#------------------------------------------------------------------------------

    def _check_required(self, k):
        """
        Checks if a key specifies a required parameter. (self._required_params)

        Parameters
        ----------
        k   :   str
            key to be checked
        """
        return k in self._required_params

#------------------------------------------------------------------------------

    def get_param_names(self):
        """

        """
        param_names = []
        for k in self.line_params.keys():
            if not (k=="x0" and isclose(self.line_params['x0'], 0.0, abs_tol=self.QUASI_TO_INELASTIC_DISTINCTION_VALUE)):
                param_names.append("_".join((k, self.name)))
        return param_names

#------------------------------------------------------------------------------

    def update_line_params(self, **new_line_params):
        """
        Updates line parameter

        Parameters
        ----------
        new_line_params : dict
            new parameter for the modeled line
        """
        # strip_param_name = lambda k: k.split("_")[0]
        # update = {strip_param_name(k) : i for k, i in new_line_params.items() if self._check_required(strip_param_name(k))}
        # self.line_params.update(update)
        self.line_params.update(new_line_params)

#------------------------------------------------------------------------------

    def update_domain(self, new_domain):
        """
        Updates line parameter

        Parameters
        ----------
        new_domain : tuple
            new domain for the modeled line
        """
        self.domain = new_domain

#------------------------------------------------------------------------------

    # def check_params(self):
    #     """
    #     Checks
    #     """
    #     try:
    #         assert set(self.line_params.keys()).issubset(set(getfullargspec(self.calc).args + ["amplitude"]))
    #     except AssertionError:
    #         raise KeyError(f"Model parameters do not match the required ones: {getfullargspec(self.calc).args[1:]}")

#------------------------------------------------------------------------------

    def normalize(self):
        pass

#------------------------------------------------------------------------------

    def within_domain(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        # return value[(value >= min(self.domain)) & (value <= max(self.domain))]
        return (value >= min(self.domain)) & (value <= max(self.domain))

#------------------------------------------------------------------------------

    @staticmethod
    def calc(var, **line_params):
        """

        """
        pass

###############################################################################
###############################################################################
###############################################################################

@LineFactory.register('lorentzian')
class LorentzianLine(Line):
    """
    A Lorentzian shaped excitation Line.
    """

    _required_params = ("x0", "width", "c", "weight")
    INTEGRATION_WIDTH_SCALE = 10000.0
    INTEGRATION_POINT_DENSITY_PARAMETER = 6000
    INTEGRATION_POINT_NUMBER_FACTOR = 2

    def normalize(self):
        """
        Calculates normalization factor of a Lorentzian Line for its domain

        Returns
        -------
        n   :   float
            Normalization factor 1 / N
        """
        # changed_sign_line_params = {k : (v if k is not "x0" else -v) for k, v in self.line_params.items()}
        # N = lorentzian_cdf(max(self.domain), **self.line_params)
        # N -= lorentzian_cdf(min(self.domain), **self.line_params)
        # N += lorentzian_cdf(max(self.domain), **changed_sign_line_params)
        # N -= lorentzian_cdf(min(self.domain), **changed_sign_line_params)
        # N += self.line_params["c"] * (max(self.domain) - min(self.domain))
        # return  1.0 / N
        return 1.0 / self.integrate()

#------------------------------------------------------------------------------

    def integrate(self):
        """
        Integratey a standard Lorentzian Line over its domain
        (Via cdf)

        Returns
        -------
        N   :   float
            Integration value
        """
        return lorentzian_cdf(max(self.domain), **self.line_params) - lorentzian_cdf(min(self.domain), **self.line_params)

#------------------------------------------------------------------------------

    @staticmethod
    def calc(var, x0, width, c, **kwargs):
        """
        Calculates a standard Lorentzian / Cauchy probability distribution.
        (normalized to 1)

        Parameters
        ----------
        x       :   float, ndarray
            running variable in our context mostly energy
        x0      :   float
            mean value of the distribution
        width   :   float
            FWHM of the distributon
        c       :   float
            constant background

        Returns
        -------
        res     :   float, ndarray
            probability distribution function results
        """
        # return lorentzian(var, x0, width) + lorentzian(var, -x0, width) + c
        return lorentzian(var, x0, width)

#------------------------------------------------------------------------------

    def get_peak_domain(self, splitted_domain=False):
        """

        """
        return (
            self.line_params["x0"] - self.INTEGRATION_WIDTH_SCALE * self.line_params["width"],
            self.line_params["x0"] + self.INTEGRATION_WIDTH_SCALE * self.line_params["width"]
        )

#------------------------------------------------------------------------------

    def export_to_dict(self):
        """
        Returns a dictionary, which can be used to create
        a new instance of the same class via the LineFactory or
        the cls.load_from_dict method as alternate constructor.

        Returns
        -------
            :   dict
            dictionary of the contents of the Line object.
        """
        export_dict = dict(
            name=self.name,
            domain=self.domain,
            **self.line_params
        )
        export_dict.update(dict(specifier="lorentzian"))
        return export_dict

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
        export_dict = dict(
            name=self.name,
            domain=self.domain,
            **self.line_params
        )
        export_dict.update(dict(specifier="lorentzian"))
        with open(json_file, "w") as jsonfile:
            json.dump(export_dict, jsonfile)
        return None

###############################################################################
###############################################################################
###############################################################################

@LineFactory.register('f_c')
class F_cLine(Line):
    """

    """

    _required_params = ("x0", "width", "A", "q", "c", "weight")
    INTEGRATION_WIDTH_SCALE = 10000.0
    INTEGRATION_POINT_DENSITY_PARAMETER = 6000
    INTEGRATION_POINT_NUMBER_FACTOR = 2

    @staticmethod
    def calc(var, x0, width, A, q, **kwargs):
        """
        Spectral shape function by Folk and Iro [1] valid at T = Tc.
        Based on renormalization-group approach by Dohm [2] for isotropic
        ferromagnets.

        Parameter
        ---------
        var             :   float, ndarray
            running variable, energy transfer [meV]
        x0              :   float
            !!! NOT NATIVE TO THEORY !!!
            shift, BY CONSTRUCTION in energy position
        width           :   float
            analogon to the linewidth in a Lorentzian spectral shape function
            Has to given in the same unit as energy variable 'e'.
        A               :   float
            proportionality factor for linewidth calculation in gamma = A * q**2.5
            [A] = meV angstroem**(5/2) | A(nickel) = 350 meV angstroem**(5/2)
        q               :   float
            momentum transfer [1/angstroem]

        Return
        ------
        fqe             :   float
            spectral shape function F(q, energy)

        References
        ----------
        [1] R. Folk and H. Iro, Phys. Rev. B 32, 1880 (1985)
        [2] V. Dohm, Solid State Commun. 20, 657 (1976)
        """
        return fqe_c((var - x0), width, A, q)

#------------------------------------------------------------------------------

    def integrate(self):
        """
        Integrate numerically over its domain

        Returns
        -------
        N   :   float
            Integration value
        """
        x = np.linspace(min(self.domain), max(self.domain), 10000)
        y = self.calc(x, **self.line_params)
        return np.trapz(y, x)

#------------------------------------------------------------------------------

    def normalize(self):
        """
        Calculates normalization factor of a Lorentzian Line for its domain

        Returns
        -------
        n   :   float
            Normalization factor 1 / N
        """
        return 1.0 / self.integrate()

#------------------------------------------------------------------------------

    def export_to_dict(self):
        """
        Returns a dictionary, which can be used to create
        a new instance of the same class via the LineFactory or
        the cls.load_from_dict method as alternate constructor.

        Returns
        -------
            :   dict
            dictionary of the contents of the Line object.
        """
        export_dict = dict(
            name=self.name,
            domain=self.domain,
            **self.line_params
        )
        export_dict.update(dict(specifier="f_c"))
        return export_dict

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
        export_dict = dict(
            name=self.name,
            domain=self.domain,
            **self.line_params
        )
        export_dict.update(dict(specifier="f_c"))
        with open(json_file, "w") as jsonfile:
            json.dump(export_dict, jsonfile)
        return None

###############################################################################
###############################################################################
###############################################################################

@LineFactory.register('f_I')
class F_ILine(Line):
    """

    """

    _required_params = ("x0", "width", "A", "q", "kappa", "c", "weight")
    INTEGRATION_WIDTH_SCALE = 10000.0
    INTEGRATION_POINT_DENSITY_PARAMETER = 6000
    INTEGRATION_POINT_NUMBER_FACTOR = 3

    @staticmethod
    def calc(var, x0, width, A, q, kappa, **kwargs):
        """
        Analytical expression for the dynamical correlation function
        of an isotropic ferromagnet in first order in
        epsilon = 6 - d dimensions introduced by Iro [1].
        Valid for T > Tc.

        Parameter
        ---------
        var             :   float, ndarray
            running variable, energy transfer [meV]
        x0              :   float
            !!! NOT NATIVE TO THEORY !!!
            shift, BY CONSTRUCTION, in energy position
        width           :   float
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
        fqI             :   float
            spectral shape function F(q, energy)

        References
        ----------
        [1] H. Iro, J. Magn. Magn. Mater. 73, 175 (1988)
        """
        return fqe_I((var - x0), width, A, q, kappa)

#------------------------------------------------------------------------------

    def integrate(self):
        """
        Integrate numerically over its domain

        Returns
        -------
        N   :   float
            Integration value
        """
        x = np.linspace(min(self.domain), max(self.domain), 10000)
        y = self.calc(x, **self.line_params)
        return np.trapz(y, x)

#------------------------------------------------------------------------------

    def normalize(self):
        """
        Calculates normalization factor of a Lorentzian Line for its domain

        Returns
        -------
        n   :   float
            Normalization factor 1 / N
        """
        return 1.0 / self.integrate()

#------------------------------------------------------------------------------

    def export_to_dict(self):
        """
        Returns a dictionary, which can be used to create
        a new instance of the same class via the LineFactory or
        the cls.load_from_dict method as alternate constructor.

        Returns
        -------
            :   dict
            dictionary of the contents of the Line object.
        """
        export_dict = dict(
            name=self.name,
            domain=self.domain,
            **self.line_params
        )
        export_dict.update(dict(specifier="f_I"))
        return export_dict

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
        export_dict = dict(
            name=self.name,
            domain=self.domain,
            **self.line_params
        )
        export_dict.update(dict(specifier="f_I"))
        with open(json_file, "w") as jsonfile:
            json.dump(export_dict, jsonfile)
        return None

###############################################################################
###############################################################################
###############################################################################

class LineCalcStrategy:
    """
    Interface for calculating a modelmiezelb.lineshape.Line
    """

    def calc(self, line):
        """

        """
        pass

#------------------------------------------------------------------------------

    def normalize(self, line):
        """

        """
        pass

###############################################################################
###############################################################################
###############################################################################

class QuasielasticCalcStrategy:
    """

    """

    def calc(self, line, var):
        """
        Evaluates a line as "quasielastic".
        Does not include the Bose factor
        result = weight * (line(var) + c) * normalization_factor

        Parameters
        ----------
        line    :   class in compliance with the modelmiezelb.lineshape.Line Interface
        var     : float, ndarray, list, tuple
            running variable of the Line

        Returns
        -------
                : numpy.ndarray
            result from evaluating the Line
        """
        return line.line_params["weight"] * (line(var) + line.line_params["c"]) * self.normalize(line) * line.within_domain(var)

#------------------------------------------------------------------------------

    def normalize(self, line):
        return 1.0 / (line.integrate() + line.line_params["c"] * (max(line.domain) - min(line.domain)))

#------------------------------------------------------------------------------

    def get_peak_domain(self, line):
        return line.get_peak_domain()

#------------------------------------------------------------------------------

    def get_adaptive_integration_grid(self, line, ne):
        """

        """
        return line.get_adaptive_integration_grid(ne)[1:]
#        return np.linspace(*self.get_peak_domain(line), npeak, endpoint=False)

###############################################################################
###############################################################################
###############################################################################

class InelasticCalcStrategy:
    """
    
    """
    def __init__(self, T):
        """

        """
        self.T = T

#------------------------------------------------------------------------------

    def calc(self, line, var):
        """
        Evaluates a line as "inelastic".
        Includes the Bose factor
        result = weight * normalization_factor * (line(var | x0) * bose_factor + line(var | -x0) + c)

        Parameters
        ----------
        line    :   class in compliance with the modelmiezelb.lineshape.Line Interface
        var     : float, ndarray, list, tuple
            running variable of the Line

        Returns
        -------
        ret     : numpy.ndarray
            result from evaluating the Line
        """
        line_dict = line.export_to_dict()
        line_dict["x0"] = abs(line_dict["x0"])
        loss_line = line.load_from_dict(**line_dict) # neutrons perspective
        line_dict["x0"] *= -1.0
        gain_line = line.load_from_dict(**line_dict) # neutrons perspective

        ret = loss_line(var) * bose_factor(abs(line.line_params["x0"]), self.T)
        ret += gain_line(var)
        ret += line.line_params["c"]
        ret *= self.normalize(loss_line)
        ret *= line.line_params["weight"]
        ret *= line.within_domain(var)
        return ret

#------------------------------------------------------------------------------

    def normalize(self, line):
        """

        """
        line_dict = line.export_to_dict()
        line_dict["x0"] = abs(line_dict["x0"])
        loss_line = line.load_from_dict(**line_dict) # neutrons perspective
        line_dict["x0"] *= -1.0
        gain_line = line.load_from_dict(**line_dict) # neutrons perspective

        retnorm = loss_line.integrate() * bose_factor(abs(loss_line.line_params["x0"]), self.T)
        retnorm += gain_line.integrate()
        retnorm += line.line_params["c"] * (max(line.domain) - min(line.domain))
        return 1.0 / retnorm

#------------------------------------------------------------------------------

    def get_peak_domain(self, line):
        """

        """
        line_dict = line.export_to_dict()
        line_dict["x0"] = abs(line_dict["x0"])
        pos_domain = line.load_from_dict(**line_dict).get_peak_domain()
        line_dict["x0"] = -1 * line_dict["x0"]
        neg_domain = line.load_from_dict(**line_dict).get_peak_domain()

        if neg_domain[1] >= pos_domain[0]:
            return (neg_domain[0], pos_domain[1])
        else:
            return neg_domain, pos_domain

#------------------------------------------------------------------------------

    def _grid(self, domain, n):
        return np.linspace(*domain, n, endpoint=False)

#------------------------------------------------------------------------------

    def get_adaptive_integration_grid(self, line, ne):
        """

        """
        ne //= 2 # Factor 1/2 to get equal amount of points from
                # inelastic and quasielastic lines.
        # First peak grid
        grid1 = line.get_adaptive_integration_grid(ne)
        # Second peak grid
        line_dict = line.export_to_dict()
        line_dict["x0"] = -1.0 * line_dict["x0"]
        grid2 = line.load_from_dict(**line_dict).get_adaptive_integration_grid(ne)
        return np.concatenate((grid1, grid2))

        # d1, d2 = self.get_peak_domain(line)
        # if isinstance(d1, tuple):
        #     return np.union1d(self._grid(d1, npeak), self._grid(d2, npeak))
        # elif isinstance(d1, float):
        #     return self._grid((d1, d2), 2*npeak)