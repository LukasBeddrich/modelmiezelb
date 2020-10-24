"""

"""

from math import isclose
from numpy import power, sum, array
from iminuit import Minuit
from os.path import join

from modelmiezelb.utils.helpers import minuit_to_dict, format_param_dict_for_logger, format_sqt_lines_for_logger
from modelmiezelb.utils.util import MIEZE_DeltaFreq_from_time

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
    def from_sqe(cls, sqe, x, y, yerr, initguess, logger=None):
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
    def from_transformer(cls, transformer, x, y, yerr, initguess, logger=None):
        """
        Creates a 'Minuit' object from `transformer´ using the '.from_array_func'
        """
        param_names = cls.get_param_names(sqe=transformer.sqemodel)
        transformer_internal = transformer.load_from_dict(**transformer.export_to_dict())

        if logger:
            def calc(params):
                params_dict = dict(zip(param_names, params))
                logger.info(f"\nParameters given to `calc´:\n{format_param_dict_for_logger(params_dict)}\n\n")
                transformer_internal.update_params(**params_dict)
                logger.info(f"\nParameters of Lines after update:\n{format_sqt_lines_for_logger(transformer_internal)}\n\n")
            
                sqtvals = array([transformer_internal(xval) for xval in x])
                cost = sum(power((y - sqtvals) / yerr, 2))
                logger.info(f"y =      {', '.join([str(val)[:7] for val in y])} \ny_theo = {', '.join([str(val)[:7] for val in sqtvals])}")
                logger.info(f"\nThe cost value was calculated to be: {cost}\n\n")
                return cost
        else:
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

#------------------------------------------------------------------------------

    @classmethod
    def _get_chi_square_from_transformer(cls, transformer, x, y, yerr):
        """
        Creates a chi-square function from a transformer object.
        """
        param_names = cls.get_param_names(sqe=transformer.sqemodel)
        transformer_internal = transformer.load_from_dict(**transformer.export_to_dict())

        def calc(params):
            params_dict = dict(zip(param_names, params))
            transformer_internal.update_params(**params_dict)
            
            sqtvals = array([transformer_internal(xval) for xval in x])
            return sum(power((y - sqtvals) / yerr, 2))

        return calc

###############################################################################
###############################################################################
###############################################################################

def simple_fit_multi_sqt_curves(transformer, contrastdata, pnames, p0s, cdidx=0, constraints=None, ncalls=150, iomanager=None, export_root=None):
    """
    Finally a function that does what my ipynb does for fitting for one particular model

    Parameters
    ----------
    transformer  :   modelmiezelb.transformer.SqtTransformer
        the theoretical model for the fit
    contrastdata :   iterable
        contains the modelmiezelb.io.ContrastData objects
    pnames       :   iterable
        contains names of the parameters in the fit
    p0s          :   list
        list of parameter lists. len(p0s) needs to equal len(contrastdata)
        len(p0s[i]) needs to equal len(pnames)
    cdidx        :   int, optional
        index of the data set within a ContrastData object
    contstraints :   dict, optional
        contraints for Minuit object
    ncalls       :   int, optional
        number of function calls in Minuit.migrad
    iomanager    :   None, modelmiezelb.io.IOManager, optional
        used for export
        !!! needs to contain JSONExporter, otherwise filenames do not make sense !!!
    export_root  :   str, optional
        directory where to save the results
        Needs to be given if iomanager is passed to function

    Return
    ------
    minuits      :   list
        list of Minuit objects after fitting

    Note
    ----
    This function assumes that MIEZE times are given in nanoseconds (ns)!
    If not the result is rubbish! 
    """
    from time import time, localtime, strftime

    try:
        assert len(contrastdata) == len(p0s)
        assert len(p0s[0]) == len(pnames)
    except:
        raise AssertionError("The dimensions of the inputs do not match!\
            Check you input for inconsistencies.")

    minuits = [None] * len(contrastdata)

    t0 = time()
    for idx in range(len(contrastdata)):
        ti = time()

        tempcd = contrastdata[idx]
        taus, c, cerr = tempcd.getData(cdidx)[[0, -2, -1]]
        freqs = MIEZE_DeltaFreq_from_time(
            taus*1.0e-9,
            transformer.params["lSD"],
            transformer.sqemodel.model_params["lam"]
        )

        # Init base Minuit object
        mbase = FitModelCreator.from_transformer(
            transformer,
            freqs,
            c,
            cerr,
            p0s[idx]
        )

        # Add constraints
        default_constraints = {"error_" + n : val * 0.006 for n, val in zip(pnames, p0s[idx])}
        if constraints:
            default_constraints.update(constraints)
        
        m = FitModelCreator.from_Minuit(
            mbase,
            **default_constraints
        )
        fmin, res = m.migrad(ncall=ncalls)
        minuits[idx] = m
        print(fmin)
        print(res)

        print("Fitting is {}% done.".format(float(idx+1)/len(contrastdata) * 100))
        print("Last iteration took {:.2f} seconds".format(time()-ti))
        print("Total seconds elapsed: {:.2f} s".format(time()-t0))
        print("The fitting step was finished at {}\n".format(strftime("%H:%M:%S  %a %Y/%m/%d", localtime())))

        tempcd.fitparams = minuit_to_dict(m)

        if bool(iomanager) and bool(export_root):
            export_path = join(export_root, f"{tempcd.get('keys')[cdidx]}")
            components = [f"_{k}{str(tempcd.get(k))}" for k in ["foilnum", "arcnum", "description"] if tempcd.get(k)]
            export_path += "".join(components) + ".json"

            iomanager.export((export_path, tempcd))

    return minuits