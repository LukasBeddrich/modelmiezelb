"""
Module for line shapes in SqE modeling.
"""

import json
import numpy as np
from types import MethodType
from inspect import getfullargspec
from .utils.lineshape_functions import lorentzian, lorentzian_cdf
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

    def update_line_params(self, **new_line_params):
        """
        Updates line parameter

        Parameters
        ----------
        new_line_params : dict
            new parameter for the modeled line
        """
        self.line_params.update(new_line_params)
#        self.check_params()

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