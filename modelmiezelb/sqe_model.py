"""
NOTE
----
Now there are two types of S(q,E) models: SqE and SqE_kf.
Maybe the implementation of a Factory class might be necessary for correct usage of fit
functionality.
"""

import json
import modelmiezelb.arg_inel_mieze_model as arg
from math import isclose
from numpy import union1d, linspace, tile, trapz, pi, ndarray, where
from scipy.integrate import quad
from .utils.util import energy_from_lambda, bose_factor, wavelength_from_energy, from_energy_to_lambdaf
from .utils.helpers import get_key_for_grouping
from .lineshape import InelasticCalcStrategy, QuasielasticCalcStrategy, LineFactory
from itertools import groupby
from functools import reduce

###############################################################################

UPPER_INTEGRATION_LIMIT = 15.0 # meV

class SqE:
    """
    A Basic model for a scattering function in momentum and energy space.
    """

    _required_params = ("lSD", "lam", "dlam", "T")

    def __init__(self, lines, **model_params):
        """
        Creates a SqE model based on a lineshape and model parameters

        Parameters
        ----------
        lines       :   list
            contains modelmiezelb.lineshape.Line (spectral) shape of the peaks
        model_params    :   dict
            contains information to build the scattering function
            lSD    : sample-detector-distance [m]
            lam     : mean wavelength of the neutrons
            dlam    : spread of the wavelength distr. (triangular)
            T       : Temperature of the sample (Bose-factor)
        """

        self._lines = lines
        self.model_params = model_params
#        self.check_model_params()
        self.inelastic = InelasticCalcStrategy(self.model_params["T"])
        self.quasielastic = QuasielasticCalcStrategy()

#------------------------------------------------------------------------------
    
    def __call__(self, var, lam=None):
        """

        """
        return self.calc(var, lam)

#------------------------------------------------------------------------------

    def get_peak_domain(self):
        """
        NOTE
        ----
        This does not consider overlapping intervals!
        Needs to be fixed for better adaptive grid initialization
        """
        peak_domains = []
        for line in self._lines:
            line_peak_domain = line.get_peak_domain()
            if abs(line_peak_domain[0]) == line_peak_domain[1]:
                peak_domains.append(line_peak_domain)
            elif line_peak_domain[0] <= line_peak_domain[1] * -1:
                absmax = max((abs(x) for x in line_peak_domain))
                peak_domains.append((-1 * absmax, absmax))
            else:
                peak_domains.append(line_peak_domain)
                peak_domains.append(tuple(sorted([-1 * d for d in line_peak_domain])))
        return peak_domains

#------------------------------------------------------------------------------

    def get_adaptive_integration_grid(self, ne, nlam):
        """
        Creates a 2D grid in energy (not yet: and wavelength) for MIEZE transform
        integration points are focused around the peaks of the Line s

        Parameters
        ----------
        ne      :   int
            round about number of energy points
        nlam    :   int
            number of points in wavelength

        Return
        ------
        ee      :   ndaray
            2D array of energy integration points
        """
        # first grid is a coarse grid over entire domain
        grids = [[]]
        for line in self._lines:
            grids.append(self.apply_grid_strategy(line, ne))
        # unites all individual grids to one.
        grid = reduce(union1d, grids)
        # truncate to the lowest accepted neutron energy
        grid = grid[grid > -0.9999 * energy_from_lambda(self.model_params["lam"] * (1 - self.model_params["dlam"]))]

        grid = tile(grid, (nlam, 1))
        return grid.T

#------------------------------------------------------------------------------

    def __add__(self, other):
        """

        """
        pass

#------------------------------------------------------------------------------

    def export_to_dict(self):
        """
        Returns a dictionary, which can be used to create
        a new instance of the same class via the 
        cls.load_from_dict method as alternate constructor.

        Returns
        -------
            :   dict
            dictionary of the contents of the Line object.
        """
        export_dict = {"model_params" : dict(**self.model_params)}
        for line in self._lines:
            try:
                assert line.name not in export_dict.keys()
                export_dict[line.name] = line.export_to_dict()
            except AssertionError:
                raise ValueError(f"The name of {line}, {line.name} is not unique.")
        return export_dict

#------------------------------------------------------------------------------

    @classmethod
    def load_from_dict(cls, **sqe_dict):
        """

        """
        model_params = sqe_dict.pop("model_params")
        for v in sqe_dict.values():
            if 'domain' in v.keys(): v['domain'] = tuple(v['domain'])
        lines = [LineFactory.create(item.pop("specifier"), **item) for item in sqe_dict.values()]
        return cls(tuple(lines), **model_params)

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
        export_dict = self.export_to_dict()
        with open(json_file, "w") as jsonfile:
            json.dump(export_dict, jsonfile)
        return None

#------------------------------------------------------------------------------

    @classmethod
    def load_from_jsonfile(cls, json_file):
        """

        """
        with open(json_file, "r") as jsonfile:
            imported_sqe = json.load(jsonfile)

        # for v in imported_sqe.values():
        #     if 'domain' in v.keys(): v['domain'] = tuple(v['domain'])
        return cls.load_from_dict(**imported_sqe)

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

    def update_params(self, **new_params):
        """

        """
        strip_param_name = lambda k: k.split("_")[0]

        grouped_params = {"model_params" : {}}
        for key, groups in groupby(new_params.items(), get_key_for_grouping):
            grouped_params[key] = dict((strip_param_name(k), v) for k, v in groups)
        self.model_params.update(grouped_params["model_params"])
        for line in self._lines:
            try:
                line.update_line_params(**grouped_params[line.name])
            except KeyError:
                pass
                # print(f"No parameter update for {line.name}")

#------------------------------------------------------------------------------

    def calc(self, var, lam=None):
        """

        """
        # retval = 0.0
        # sum_of_weights = 0.0
        # for line in self._lines:
        #     if isclose(line.line_params["x0"], 0.0, abs_tol=0.001):
        #         retval += self.quasielastic.calc(line, var)
        #         sum_of_weights += line.line_params["weight"]
        #     else:
        #         retval += self.inelastic.calc(line, var)
        #         sum_of_weights += line.line_params["weight"]
        # return retval / sum_of_weights

        retval = 0.0
        sum_of_weights = 0.0
        for line in self._lines:
            retval += self.apply_calc_strategy(line, var)
            sum_of_weights += line.line_params["weight"]
        return retval / sum_of_weights

#------------------------------------------------------------------------------

    def apply_calc_strategy(self, line, var):
        """

        """
        if isclose(line.line_params["x0"], 0.0, abs_tol=line.QUASI_TO_INELASTIC_DISTINCTION_VALUE):
            return self.quasielastic.calc(line, var)
        else:
            return self.inelastic.calc(line, var)

#------------------------------------------------------------------------------

    def apply_grid_strategy(self, line, npeak=1000):
        """

        """
        if isclose(line.line_params["x0"], 0.0, abs_tol=line.QUASI_TO_INELASTIC_DISTINCTION_VALUE):
            return self.quasielastic.get_adaptive_integration_grid(line, npeak)
        else:
            return self.inelastic.get_adaptive_integration_grid(line, npeak)

#------------------------------------------------------------------------------

    # def check_model_params(self):
    #     """

    #     """
    #     reference_set = {"lam", "dlam", "lSD", "T"}
    #     diff_set = reference_set.difference(set(self.model_params.keys()))
    #     if diff_set:
    #         raise KeyError(f"No values for model parameters '{diff_set}' were given!")

#------------------------------------------------------------------------------

    def plot(self, npoints=3001):
        """

        """
        import matplotlib.pyplot as plt
        from numpy import linspace

        try:
            assert all([l.domain == self._lines[0].domain for l in self._lines])
        except AssertionError:
            raise AssertionError("The domains of the lines do not match.\nThis is not properly caught, yet.")
        var = linspace(-UPPER_INTEGRATION_LIMIT, UPPER_INTEGRATION_LIMIT, npoints)
        
        plt.plot(var, self.calc(var))
        plt.show()

#------------------------------------------------------------------------------

    def update_domain(self, new_domain):
        """
        Updates line parameter

        Parameters
        ----------
        new_domain : tuple
            new domain for the modeled line
        """
        for line in self._lines:
            line.update_domain(new_domain)

#------------------------------------------------------------------------------

    # def build(self):
    #     """

    #     """
    #     pass

#------------------------------------------------------------------------------

    # def normalize(self):
    #     """

    #     """
    #     pass

###############################################################################
###############################################################################
###############################################################################

class SqE_kf(SqE):
    """

    NOTE
    ----
    - Add the kf/ki factor for the double differential cross section (ddcs)
    - Introduce it on sqe level --> lam0 needs to be known
                                --> only multiply once
                                --> requires numerical intergration for normalization
    """


    def __call__(self, de, lam=None):
        """
        Calculates the normalized SqE
        
        Parameters
        ----------
        de      :   float, ndarray
            the energy transfer
        lam     :   float, ndarray, NoneType
            wavelength of the incoming neutrons
        """
        return self.calc(de, lam)

#------------------------------------------------------------------------------

    def calc(self, de, lam=None):
        """
        Calculates the normalized SqE
        
        Parameters
        ----------
        de      :   float, ndarray
            the energy transfer
        lam     :   float, ndarray, NoneType
            wavelength of the incoming neutrons
        """

        return self._calc(de, lam) / self._integrate(lam)

#------------------------------------------------------------------------------

    def _calc(self, de, lam):
        """
        Calculates the non-normalized SqE

        Parameters
        ----------
        de      :   float, ndarray
            the energy transfer
        lam     :   float, ndarray, NoneType
            wavelength of the incoming neutrons
        """

        retval = 0.0
        sum_of_weights = 0.0
        # make sure, incoming neutron wavelength corresponds to the maximum energy loss
        if lam is None:
            kf = 2 * pi / from_energy_to_lambdaf(de, self.model_params["lam"])
        else:
            kf = 2 * pi / from_energy_to_lambdaf(de, lam)
        # Iterate over all lines and add up their contribution
        for line in self._lines:
            retval += self.apply_calc_strategy(line, de)
            # Add respective weight of the line to normalize sqe afterwards
            sum_of_weights += line.line_params["weight"]
        # multiply with the kf weighting factor
        retval *= kf
        return retval / sum_of_weights

#------------------------------------------------------------------------------

    def _integrate(self, lam=None):
        """
        NOTE
        ----
        The arbitrary chosen integration boundaries are a huge problem!
        Needs to be fixed with a general solution SOON!

        Parameters
        ----------
        de      :   float, ndarray
            the energy transfer
        lam     :   float, ndarray, NoneType
            wavelength of the incoming neutrons
        """
        # generate new energy array if nothing is specified
        # ee = self.get_adaptive_integration_grid(2000, 1)

        if lam is None:
            ee = self.get_adaptive_integration_grid(2000, 1)
            ee = ee[ee>-0.99999*energy_from_lambda(self.model_params["lam"])]
            ll = lam
        else:
            ee = self.get_adaptive_integration_grid(2000, lam.shape[1])
            ll = tile(lam[0], (ee.shape[0], 1))
            a = -0.99999*energy_from_lambda(ll)
            ee = where(ee <= a, a, ee)
        return trapz(self._calc(ee, ll), ee, axis=0)

###############################################################################
###############################################################################
###############################################################################

class SqE_from_arg:
    def __init__(self, **model_params):
        self.model_params = dict(lam=6.0, lSD=3.43, dlam=0.12)
        self.model_params.update(model_params)

#------------------------------------------------------------------------------

    def update_domain(self, new_domain):
        pass

#------------------------------------------------------------------------------

    def __call__(self, var, T=20.0):
        return arg.SvqE(func=arg.fqe_I,
            e=var,
            e0=7.0,
            e0width=0.2,
            bckg=0.0,
            amplitude=1.0,
            lam=self.model_params["lam"],
            T=T,
            q=0.00005,
            kappa=0.1,
            A=350.
        )

#------------------------------------------------------------------------------

    def get_adaptive_integration_grid(self, ne, nlam):
        """
        Mimiks SqE get_adaptive_integration_grid method

        Parameters
        ----------
        ne      :   int
            round about number of energy points
        nlam    :   int
            number of points in wavelength

        Return
        ------
        ee      :   ndaray
            2D array of energy integration points
        """
        from modelmiezelb.lineshape import F_ILine
        line = F_ILine(
            "FI1",
            domain=(-15.0, 15.0),
            x0=0.03,
            width=0.02,
            A=367.0,
            q=0.015,
            kappa=0.005,
            c=0.0,
            weight=1.0
        )
        sqe = SqE((line,), lam=6.0, dlam=0.12, lSD=3.43, T=610.0)
        return sqe.get_adaptive_integration_grid(ne, nlam)

###############################################################################
###############################################################################
###############################################################################

# class SqE_combined(SqE):
#     """
#     A S(q,E) model containing quasielastic and inelastic lines.
#     """
        
#     def calc(self, var):
#         """
#         TODO
#         ----
#         - fix this by temporarily create a line, with another Line obj
#         - fix inconsistency with adding a constant background
#             (let a line know that it is inelastic?)
#         - cut-off in bose_factor?
#         """
#         retval = 0.0
#         sum_of_amplitudes = 0.0
#         for line in self._lines:
#             line.update_domain(
#                 (-0.999 * energy_from_lambda(self.model_params["lam"]), UPPER_INTEGRATION_LIMIT)
#             )
#             retval += line.line_params["amplitude"] * line.normalize() * line(var)
#             sum_of_amplitudes += line.line_params["amplitude"]
        
#         # Account fr amplitude weighting
#         return retval / sum_of_amplitudes