"""

"""
import json
from numpy import tile, linspace, empty, cos, trapz, meshgrid, ones, pi, sqrt, where, atleast_2d
from modelmiezelb import resdir
from modelmiezelb.utils.util import energy_from_lambda, MIEZE_phase, triangle_distribution, detector_efficiency
from modelmiezelb.sqe_model import UPPER_INTEGRATION_LIMIT, SqE
from modelmiezelb.correction import CorrectionFactor


class Transformer:
    """

    """

    def __init__(self):
        """

        """
        pass

#------------------------------------------------------------------------------

    def __call__(self, var):
        """

        """
        pass

#------------------------------------------------------------------------------

    def calc(self, var):
        """

        """
        pass

#------------------------------------------------------------------------------

    def export_to_jsonfile(self, json_file):
        """

        """
        pass

#------------------------------------------------------------------------------

    def load_from_jsonfile(self, json_file):
        """

        """
        pass

###############################################################################
###############################################################################
###############################################################################

class SqtTransformer(Transformer):
    """

    """

    _required_params = ("ne", "nlam", "lSD", "integ_mode")

    def __init__(self, sqemodel, corrections=(), **params):
        """
        Instantiates a Transformer to calculate the Sqt curve
        from a SqE model.

        Parameters
        ----------
        sqemodel    :   modelmiezelb.sqe_model.SqE
            A SqE model
        corrections :   tuple of a modelmiezelb.corrections.CorrectionFactor
        params      :   dict
            Additional parameters necessary for clacluation
        """
        self.sqemodel = sqemodel
        self.corrections = corrections

        self.params = dict(ne=15000, nlam=50, lSD=None, integ_mode="adaptive")
        self.params.update(params)

#------------------------------------------------------------------------------

    def __call__(self, var):
        return self.calc(var)

#------------------------------------------------------------------------------

    def calc(self, var):
        """
        Parameters
        ----------
        var     :   float
            in this case freq, the MIEZE (difference) frequency of the MIEZE coils

        Return
        ------
                :   float
            S(q,t) value
        """
        ### Creating energy, wavelength parameter space
        lam = self.sqemodel.model_params["lam"]
        dlam = self.sqemodel.model_params["dlam"]

        l = linspace(1-dlam*1.01, 1+dlam*1.01, self.params["nlam"]) * lam
        a = -0.99999 * energy_from_lambda(l)
        # Selecting integration mode
        if self.params["integ_mode"] == "adaptive":
#            print("adaptive grid")
            ee = self.sqemodel.get_adaptive_integration_grid(
                self.params["ne"],
                self.params["nlam"]
            )
#            return ee
            ee = where(ee <= atleast_2d(a), atleast_2d(a), ee)

        elif self.params["integ_mode"] == "linear":
#            print("linear grid")
            ee = linspace(a, UPPER_INTEGRATION_LIMIT, self.params["ne"])
#            return ee

        ne = ee.shape[0]
        ll = tile(l,(ne, 1))
#        return ee, ll

        ### Creating intgrand arrays
        sqe = ones((ne, self.params["nlam"]))
        corrs = tile(sqe, (len(self.corrections), 1, 1))

        ### Filling integrand arrays
        mieze_phase = MIEZE_phase(ee, var, self.params["lSD"], ll)
        det_eff = detector_efficiency(ee, ll, 1)
        tri_distr = triangle_distribution(ll, lam, dlam)
        for cidx, cf in enumerate(self.corrections):
            corrs[cidx] = cf(ee, ll)
        corrs = corrs.prod(axis=0)

        for lamidx, clam in enumerate(l): # clam = current wavelength
            self.sqemodel.update_domain((-1 * energy_from_lambda(clam), UPPER_INTEGRATION_LIMIT))
            sqe[:, lamidx] = self.sqemodel(ee[:, lamidx])

        ### Integration with trapezoidal rule
        # integrand except phase
        integrand = det_eff * sqe * tri_distr * corrs
        # integration for two points of the of known phase difference
        y1 = trapz(trapz(integrand * cos(mieze_phase), ee, axis=0), l)
        y2 = trapz(trapz(integrand * cos(mieze_phase + 0.5*pi), ee, axis=0), l)
        return ( y1**2 + y2**2 )**0.5

#------------------------------------------------------------------------------

    def export_to_dict(self):
        """
        Returns a dictionary, which can be used to create
        a new instance of the same class via the 
        cls.load_from_dict method as alternate constructor.
        
        !DOES NOT WORK WITH CORRECTIONS YET!

        Returns
        -------
            :   dict
            dictionary of the contents of the Transformer object.
        """
        export_dict = dict(params=self.params)
        export_dict['corrections'] = [corr.export_to_dict() for corr in self.corrections]
        export_dict['sqemodel'] = self.sqemodel.export_to_dict()
        return export_dict

#------------------------------------------------------------------------------

    @classmethod
    def load_from_dict(cls, **transformer_dict):
        """

        """
        params = transformer_dict["params"]
        sqemodel = SqE.load_from_dict(**transformer_dict["sqemodel"])
        corrections = tuple([CorrectionFactor.load_from_dict(**cdict) for cdict in transformer_dict["corrections"]])
        return cls(sqemodel, corrections, **params)

#------------------------------------------------------------------------------

    def export_to_jsonfile(self, json_file):
        """
        Saves a JSON-file, which can be used to create
        a new instance of the same class via the 
        cls.load_from_jsonfile method as alternate constructor.
        
        !DOES NOT WORK WITH CORRECTIONS YET!

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

    @classmethod
    def load_from_jsonfile(cls, json_file):
        """

        """
        with open(json_file, "r") as jsonfile:
            imported_transformer = json.load(jsonfile)

        for v in imported_transformer["sqemodel"].values():
            if 'domain' in v.keys(): v['domain'] = tuple(v['domain'])
        return cls.load_from_dict(**imported_transformer)

#------------------------------------------------------------------------------

    def update_params(self, **new_params):
        """

        """
        transformer_specific_keys = set(self._required_params).intersection(set(new_params.keys()))
        self.params.update({k : new_params[k] for k in transformer_specific_keys})
        for corr in self.corrections:
            corr.update_params(**new_params)
        self.sqemodel.update_params(**new_params)
        
#------------------------------------------------------------------------------

###############################################################################
###############################################################################
###############################################################################

# A SqtTransformer object for everyday usage
STANDARD_SQT = SqtTransformer.load_from_jsonfile(f"{resdir}/std_sqt.json")
