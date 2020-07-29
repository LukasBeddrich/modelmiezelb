"""

"""

import json
import modelmiezelb.arg_inel_mieze_model as arg
from math import isclose
from .utils.util import energy_from_lambda, bose_factor, wavelength_from_energy
from .lineshape import InelasticCalcStrategy, QuasielasticCalcStrategy

###############################################################################

UPPER_INTEGRATION_LIMIT = 15.0 # meV

class SqE:
    """
    A Basic model for a scattering function in momentum and energy space.
    """

    def __init__(self, lines, **model_params):
        """
        Creates a SqE model based on a lineshape and model parameters

        Parameters
        ----------
        lines       :   list
            contains modelmiezelb.lineshape.Line (spectral) shape of the peaks
        model_params    :   dict
            contains information to build the scattering function
            l_SD    : sample-detector-distance [m]
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
    
    def __call__(self, var):
        """

        """
        return self.calc(var)

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
        export_dict = dict(**self.model_params)
        for line in self._lines:
            try:
                assert line.name not in export_dict.keys()
                export_dict[line.name] = line.export_to_dict()
            except AssertionError:
                raise ValueError(f"The name of {line}, {line.name} is not unique.")
        return export_dict

#------------------------------------------------------------------------------

    def export_to_jsonfile(self, json_file):
        """

        """
        export_dict = self.export_to_dict()
        with open(json_file, "w") as jsonfile:
            json.dump(export_dict, jsonfile)
        return None

#------------------------------------------------------------------------------

    def calc(self, var):
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
        if isclose(line.line_params["x0"], 0.0, abs_tol=0.001):
            return self.quasielastic.calc(line, var)
        else:
            return self.inelastic.calc(line, var)

#------------------------------------------------------------------------------

    # def check_model_params(self):
    #     """

    #     """
    #     reference_set = {"lam", "dlam", "l_SD", "T"}
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

class SqE_from_arg:
    def __init__(self, **model_params):
        self.model_params = dict(lam=6.0, l_SD=3.43, dlam=0.12)
        self.model_params.update(model_params)

#------------------------------------------------------------------------------

    def update_domain(self, new_domain):
        self.model_params["lam"] = 6.0#wavelength_from_energy(abs(new_domain[0]))

#------------------------------------------------------------------------------

    def __call__(self, var):
        return arg.SvqE(func=arg.fqe_I,
            e=var,
            e0=-1.0,
            e0width=0.4,
            bckg=0.0,
            amplitude=1.0,
            lam=6.0,#self.model_params["lam"],
            T=20,
            q=0.00005,
            kappa=0.1,
            A=350.
        )

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