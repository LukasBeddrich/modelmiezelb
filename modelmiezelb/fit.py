"""

"""

from math import isclose
from numpy import power, sum, array
from iminuit import Minuit
from .utils.helpers import flatten_list

class LeastSquareCollector_old: #ParamCollector
    """
    Inspects a model and recoveres fit parameters.
    """
    @classmethod
    def from_sqe(cls, sqe):
        param_names = {}
        for l in sqe._lines:
            names = []
            for k in l.line_params.keys():
                if not (k=="x0" and isclose(l.line_params['x0'], 0.0, abs_tol=0.001)):
                    names.append(k)
            param_names[l.name] = names
        
        funcstr = cls.create_func(**param_names)
        return funcstr

    @classmethod
    def from_transformer(cls, transformer):
        pass

    @classmethod
    def create_func(cls, **param_names):
        renamed_params = [['_'.join((v, k)) for v in vals] for k, vals in param_names.items()]
        calc_code = f"""def resids(self, var, {", ".join(list(flatten_list(renamed_params)))}):\n"""
        return calc_code
        calc_code_body = [
            "\tfor l in self._lines:",
            f"""\t\tself.line.line_params.update({"dict(" + ", ".join([f"{k}={'_'.join((v, k))}" for k, v in param_names.items()]) + ")"})""",
            "pass"
        ]
        calc_code += "\n".join(calc_code_body)
        return calc_code

###############################################################################
###############################################################################
###############################################################################

# class Fitter:
#     """

#     """
#     def __init__(self):
#         pass

#     def _create_calc(self):
        

#         def calc(params):
#             ### UPDATE
#             ### DO the model calculation
#             return ymodel

###############################################################################
###############################################################################
###############################################################################

class FitModelCreator:
    """

    """
    @classmethod
    def get_param_names(cls, sqe=None, minuitobj=None):
        """

        """
        if sqe:
            param_names = []
            for l in sqe._lines:
                for k in l.line_params.keys():
                    if not (k=="x0" and isclose(l.line_params['x0'], 0.0, abs_tol=0.001)):
                        param_names.append("_".join((k, l.name)))
            return param_names

        elif minuitobj:
            param_names = [name for name in minuitobj.fitarg.keys() if name.split("_")[0] not in ['fix', 'limit', 'error']]
            return param_names

        else:
            return []

#------------------------------------------------------------------------------

    @classmethod
    def from_sqe(cls, sqe, x, y, yerr, initguess):
        """
        Creates a 'Minuit' object from sqe model using the '.from_array_func'
        """
        param_names = []
        for l in sqe._lines:
            for k in l.line_params.keys():
                if not (k=="x0" and isclose(l.line_params['x0'], 0.0, abs_tol=0.001)):
                    param_names.append("_".join((k, l.name)))

        def calc(params):
            sqe_internal = sqe.load_from_dict(**sqe.export_to_dict())
            params_dict = dict(zip(param_names, params))
            for line in sqe_internal._lines:
                # print(line)
                # print({k.split("_")[0] : v for k, v in params_dict.items() if k.split("_")[1] == line.name})
                line.line_params.update({k.split("_")[0] : v for k, v in params_dict.items() if k.split("_")[1] == line.name})
            return sum(power((y - sqe_internal(x)) / yerr, 2))
#        return calc, param_names
        return Minuit.from_array_func(calc, initguess, name=param_names, errordef=1)

#------------------------------------------------------------------------------

    @classmethod
    def from_transformer(cls, transformer, x, y, yerr, initguess):
        """
        Creates a 'Minuit' object from `transformer´ using the '.from_array_func'
        """
        param_names = []
        for l in transformer.sqemodel._lines:
            for k in l.line_params.keys():
                if not (k=="x0" and isclose(l.line_params['x0'], 0.0, abs_tol=0.001)):
                    param_names.append("_".join((k, l.name)))

        def calc(params):
            transformer_internal = transformer.load_from_dict(**transformer.export_to_dict())
            params_dict = dict(zip(param_names, params))
            for line in transformer_internal.sqemodel._lines:
                # print(line)
                # print({k.split("_")[0] : v for k, v in params_dict.items() if k.split("_")[1] == line.name})
                line.line_params.update({k.split("_")[0] : v for k, v in params_dict.items() if k.split("_")[1] == line.name})
            sqtvals = array([transformer_internal(xval) for xval in x])
            return sum(power((y - sqtvals) / yerr, 2))
        return Minuit.from_array_func(calc, initguess, name=param_names, errordef=1)

#------------------------------------------------------------------------------

    @classmethod
    def from_Minuit(cls, minuitobj, **kwargs):
        """

        """
        fitargs = minuitobj.fitarg
        kwargs_for_update = {k : v for k, v in kwargs.items() if k in fitargs.keys()}
        fitargs.update(kwargs_for_update)
        param_names = cls.get_param_names(minuitobj=minuitobj)
        start = [fitargs[param] for param in param_names]

        return Minuit.from_array_func(
            minuitobj.fcn, start,
            name=param_names,
            errordef=1,
            **fitargs
        )



# def model(self, var, x0_1, width_1, c_1, weight_1, ..., x0_n, ... weight_n):
#     pass