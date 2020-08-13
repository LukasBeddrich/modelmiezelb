"""

"""

from math import isclose
from numpy import power, sum, array
from iminuit import Minuit

###############################################################################

class FitModelCreator:
    """
    Bundles functionality to create Minuit objects for fitting.
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
        Creates a 'Minuit' object from `SqE´ model using the '.from_array_func'
        """
        # param_names = []
        # for l in sqe._lines:
        #     for k in l.line_params.keys():
        #         if not (k=="x0" and isclose(l.line_params['x0'], 0.0, abs_tol=0.001)):
        #             param_names.append("_".join((k, l.name)))
        param_names = cls.get_param_names(sqe=sqe)

        def calc(params):
            sqe_internal = sqe.load_from_dict(**sqe.export_to_dict())
            params_dict = dict(zip(param_names, params))
            for line in sqe_internal._lines:
                line.line_params.update({k.split("_")[0] : v for k, v in params_dict.items() if k.split("_")[1] == line.name})
            return sum(power((y - sqe_internal(x)) / yerr, 2))
        return Minuit.from_array_func(calc, initguess, name=param_names, errordef=1)

#------------------------------------------------------------------------------

    @classmethod
    def from_transformer(cls, transformer, x, y, yerr, initguess):
        """
        Creates a 'Minuit' object from `transformer´ using the '.from_array_func'
        """
        param_names = cls.get_param_names(sqe=transformer.sqemodel)
        transformer_internal = transformer.load_from_dict(**transformer.export_to_dict())

        def calc(params):
            params_dict = dict(zip(param_names, params))
            transformer_internal.update_params(**params_dict)
            
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