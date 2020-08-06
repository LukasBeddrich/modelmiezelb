import os
import matplotlib.pyplot as plt
import numpy as np
###############################################################################
from modelmiezelb.sqe_model import SqE, SqE_from_arg, UPPER_INTEGRATION_LIMIT
from modelmiezelb.lineshape import Line, LorentzianLine
from modelmiezelb.utils.util import energy_from_lambda
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
    sqe = SqE(lines=(L1, L2), lam=6.0, dlam=0.12, l_SD=3.43, T=20)
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

#------------------------------------------------------------------------------

def test_sqe_arg():
    sqe_arg = SqE_from_arg(T=20.)

#------------------------------------------------------------------------------

def test_export_load():
    # We need some lines
    L1, L2 = we_need_some_Lines()
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2), lam=6.0, dlam=0.12, l_SD=3.43, T=20)
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

if __name__ == "__main__":
#    test_sqe_normalization()
#    test_sqe_arg()
    test_export_load()