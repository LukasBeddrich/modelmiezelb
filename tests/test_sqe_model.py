import os
import matplotlib.pyplot as plt
import numpy as np
###############################################################################
from modelmiezelb.sqe_model import SqE, SqE_from_arg, UPPER_INTEGRATION_LIMIT
from modelmiezelb.lineshape import Line, LorentzianLine, F_ILine
from modelmiezelb.utils.util import energy_from_lambda
###############################################################################
from pprint import pprint
###############################################################################
# Path quarrels
testdir = os.path.dirname(os.path.abspath(__file__))

def we_need_some_Lines():
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=2)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-1.0, width=0.4, c=0.02, weight=1)
    return L1, L2

#------------------------------------------------------------------------------

def test_build_sqe():
    pass

#------------------------------------------------------------------------------

def test_sqe_normalization():
    # We need some lines
    L1, L2 = we_need_some_Lines()
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    new_domain = (-1 * energy_from_lambda(6.0), UPPER_INTEGRATION_LIMIT)
    sqe.update_domain(new_domain)

    # integrate over domain
    from scipy.integrate import quad

    to_integrate = lambda x: sqe(x)
    valdom, errdom = quad(to_integrate, *new_domain)
    valover, errover = quad(to_integrate, -15, 20)
    print(f"Integration value over domain from {new_domain[0]:.5f} to {UPPER_INTEGRATION_LIMIT}: {valdom:.5f} +- {errdom:.5f}")#   |   normalization factor: {n:.5f}")
    print(f"Integration value beyond domain from -15.0 to 20.0: {valover:.5f} +- {errover:.5f}")
#    print(f"Normalized Sqe area: {val*n}")

    # - - - - - - - - - - - - - - - - - - - -

    L3 =  F_ILine("F_I1", (-energy_from_lambda(6.0), 15), x0=-0.1, width=0.008, A=350.0, q=0.02, kappa=0.01, c=0.0, weight=1)
    L4 = LorentzianLine("Lorentzian1", (-energy_from_lambda(6.0), 15), x0=-0.3, width=0.008, c=0.0, weight=1)
    sqecombined = SqE(lines=(L3, L4), lam=6.0, dlam=0.12, lSD=3.43, T=628.0)
    valdom, errdom = quad(sqecombined, *new_domain)
    print(f"Integration value over domain from {new_domain[0]:.5f} to {UPPER_INTEGRATION_LIMIT}: {valdom:.5f} +- {errdom:.5f}")#"   |   normalization factor: {n:.5f}")
    sqecombined.update_domain((-5, 5))
    valover, errover = quad(sqecombined, -2, 2)
    print(f"Integration value beyond domain from -2.0 to 2.0: {valover:.5f} +- {errover:.5f}")

#------------------------------------------------------------------------------

def test_sqe_arg():
    sqe_arg = SqE_from_arg(T=20.)

#------------------------------------------------------------------------------

def test_export_load():
    # We need some lines
    L1, L2 = we_need_some_Lines()
    L3 =   F_ILine("F_I1", (-energy_from_lambda(6.0), 15), x0=-0.1, width=0.008, A=350.0, q=0.02, kappa=0.01, c=0.0, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2, L3), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    new_domain = (-1 * energy_from_lambda(6.0), UPPER_INTEGRATION_LIMIT)
    sqe.update_domain(new_domain)

    # export to file
    sqe.export_to_jsonfile(f"{testdir}/resources/test_sqe_export_load_file.json")
    sqe_dict = sqe.export_to_dict()
    # import again
    sqe_loaded_from_dict = sqe.load_from_dict(**sqe_dict)
    sqe_loaded_from_file = sqe.load_from_jsonfile(f"{testdir}/resources/test_sqe_export_load_file.json")
    
    # # visualize
    # e = np.linspace(-15.0, 15.0, 301)
    # plt.plot(e, sqe(e), color="C0", lw=5.0, label="original")
    # plt.plot(e, sqe_loaded_from_dict(e), color="C1", ls="--", lw=2.0, label="from dict")
    # plt.plot(e, sqe_loaded_from_file(e), color="C3", ls=":", lw=2.0, label="from file")
    # plt.legend()
    # plt.show()

#------------------------------------------------------------------------------

def test_update_params():
    # We need some lines
    L1, L2 = we_need_some_Lines()
    L3 =   F_ILine("FI1", (-energy_from_lambda(6.0), 15), x0=-0.1, width=0.008, A=350.0, q=0.02, kappa=0.01, c=0.0, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2, L3), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    pprint(sqe.export_to_dict())
    tdict = dict(
        T=628.0,
#        lam=8.0,
        x0_Lorentzian2=-0.5,
        weight_Lorentzian1=5,
        kappa_FI1=0.065
    )
    sqe.update_params(**tdict)
    pprint(sqe.export_to_dict())

#------------------------------------------------------------------------------

def test_get_peak_domain():
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.2, c=0.0, weight=2)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-1.2, width=0.1, c=0.02, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    print("L1 peak domain: ", L1.get_peak_domain())
    print("L2 peak domain: ", L2.get_peak_domain())
    print("sqe peak domain: ", sqe.get_peak_domain())

#------------------------------------------------------------------------------

def test_get_adaptive_integration_grid():
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.2, c=0.0, weight=2)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-0.5, width=0.07, c=0.0, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)


    print(sqe.quasielastic.get_adaptive_integration_grid(L1, 10))
    print(sqe.inelastic.get_adaptive_integration_grid(L2, 5))
    print(sqe.get_adaptive_integration_grid(npeak=5, nrest=10))

#------------------------------------------------------------------------------

if __name__ == "__main__":
#    test_sqe_normalization()
#    test_sqe_arg()
#    test_export_load()
    test_update_params()
#    test_get_peak_domain()
#    test_get_adaptive_integration_grid()