from modelmiezelb.sqe_model import SqE, SqE_from_arg, UPPER_INTEGRATION_LIMIT
from modelmiezelb.lineshape import Line, LorentzianLine
from modelmiezelb.utils.util import energy_from_lambda

def test_build_sqe():
    pass

def test_sqe_normalization():
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=2)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-1.0, width=0.4, c=0.02, weight=1)
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

def test_sqe_arg():
    sqe_arg = SqE_from_arg(T=20.)
    

if __name__ == "__main__":
#    test_sqe_normalization()
    test_sqe_arg()