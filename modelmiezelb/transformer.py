"""

"""
from numpy import tile, linspace, empty, cos, trapz, meshgrid, ones, pi, sqrt
from modelmiezelb.utils.util import energy_from_lambda, MIEZE_phase, triangle_distribution, detector_efficiency
from modelmiezelb.sqe_model import UPPER_INTEGRATION_LIMIT
class Transformer:
    """

    """

    def __init__(self):
        """

        """
        pass

    def __call__(self, var):
        """

        """
        pass

    def calc(self, var):
        """

        """
        pass

    def load_from_jsonfile(self, json_file):
        """

        """
        pass

    def export_to_jsonfile(self, json_file):
        """

        """
        pass

###############################################################################
###############################################################################
###############################################################################

class SqtTransformer(Transformer):
    """

    """

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

        self.params = dict(n_e=15000, n_lam=50, l_SD=None)
        self.params.update(params)

    def __call__(self, var):
        return self.calc(var)

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

        l = linspace(1-dlam*1.01, 1+dlam*1.01, self.params["n_lam"]) * lam
        ll = tile(l,(self.params["n_e"],1))
        a = -0.99999 * energy_from_lambda(l)
        ee = linspace(a, UPPER_INTEGRATION_LIMIT, self.params["n_e"])

        ### Creating intgrand arrays
        sqe = ones((self.params["n_e"], self.params["n_lam"]))
        corrs = tile(sqe, (len(self.corrections), 1, 1))

        ### Filling integrand arrays
        mieze_phase = MIEZE_phase(ee, var, self.params["l_SD"], ll)
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