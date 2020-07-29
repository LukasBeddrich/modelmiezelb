import numpy as np
import pytest
import matplotlib.pyplot as plt
import os
#------------------------------------------------------------------------------
from modelmiezelb.utils.lineshape_functions import gaussian, lorentzian, fqe_I
from modelmiezelb.utils.util import energy_from_lambda
from modelmiezelb.lineshape import Line, LorentzianLine, LineFactory
#------------------------------------------------------------------------------
# Path quarrels
testdir = os.path.dirname(os.path.abspath(__file__))
print(testdir)

#------------------------------------------------------------------------------


def test_fqe_I():
    e = np.linspace(-1.5, 1.5, 151)     # meV
    e_c = 0.2                           # meV
    A = 350.0                           # meV A**2.5
    q = 0.05                            # 1/A
    kappa = 0.021                       # 1/A

    pdf1 = fqe_I(e, e_c, A, q, kappa)
    pdf2 = fqe_I(e, e_c, A, q, kappa=0.001)
    pdf3 = fqe_I(e, e_c, A, q, kappa=1.0)
    pdf4 = fqe_I(e, e_c, A, q, kappa=10.0)

    # plt.plot(e, pdf1, label=f"$\\kappa$ = {kappa}")
    # plt.plot(e, pdf2, label=f"$\\kappa$ = {0.001}")
    # plt.plot(e, pdf3, label=f"$\\kappa$ = {1.0}")
    # plt.plot(e, pdf4, label=f"$\\kappa$ = {10.0}")
    # plt.legend(loc="upper left")
    # plt.show()

#------------------------------------------------------------------------------

def test_Lines():
    l1 = Line("LineBase", (-15., 15), x0=0, FWHM=0.5)
    # assert isinstance(l1, Line)
    # try:
    #     l1.check_params()
    # except KeyError:
    #     print("KeyError was caught and 'handled' in this test.")

    l2 = LorentzianLine("Lorentzian1", (-15., 15.), x0=0.2, width=0.3)
#    l2.check_params()
    l2.update_line_params(c=0.2)

    e = np.linspace(-1.5, 1.5, 16)
    print("Returned by 'l1.calc': ", l1.calc(e, **l1.line_params))
    print("Returned by 'l2(e)': ", l2(e))

#------------------------------------------------------------------------------

def test_Lorentzian_normalization():
    Lorentzian = LorentzianLine("Lorentzian1", (-5., 5), x0=0.0, width=0.5, c=0.0)
    n = Lorentzian.normalize()

    from scipy.integrate import quad
    to_integrate = lambda x: Lorentzian(x)
    val, err = quad(to_integrate, min(Lorentzian.domain), max(Lorentzian.domain))
    print(f"Integration value: {val:.5f} +- {err:.5f}   |   normalization factor: {n:.5f}")
    print(f"Normalized Line area: {val*n}")

#------------------------------------------------------------------------------

def test_domain_enforcement_visually():

    Lorentzian_short = LorentzianLine("Lorentzian_short", (-energy_from_lambda(6.0 * 0.88), 15), x0=-0.3, width=0.5, c=0.2)
    Lorentzian_mid = LorentzianLine("Lorentzian_mid", (-energy_from_lambda(6.0), 15), x0=-0.3, width=0.5, c=0.2)
    Lorentzian_long = LorentzianLine("Lorentzian_long", (-energy_from_lambda(6.0 * 1.12), 15), x0=-0.3, width=0.5, c=0.2)

    e = np.linspace(-10.0, 20.0, 1201)

    plt.plot(e, Lorentzian_short(e))
    plt.plot(e, Lorentzian_mid(e), ls="--", lw=2.0)
    plt.plot(e, Lorentzian_long(e), ls="dotted", lw=4.0)
#    plt.show()

#------------------------------------------------------------------------------

def test_domain_enforcement():
    Lorentzian = LorentzianLine("Lorentzian1", (-5., 5), x0=-0.3, width=0.5, c=0.2)

    e_in_domain = np.linspace(-5.0, 5.0, 11)
    e_beyond_domain = np.linspace(-10.0, 10.0, 21)

    p1 = Lorentzian(e_in_domain)
    p2 = Lorentzian(e_beyond_domain)

    assert np.all(p1 == p2[Lorentzian.within_domain(e_beyond_domain)])

#------------------------------------------------------------------------------

def test_creating_line_with_LineFactory():
    L = LineFactory.create(
        'lorentzian',
        name="Lorentzian",
        domain=(-5.0, 5.0),
        x0=-0.5,
        width=0.4,
        c=0.2
    )
    assert isinstance(L, LorentzianLine)

#------------------------------------------------------------------------------

def test_export_load():
    L = LorentzianLine(
        name="Lorentzian",
        domain=(-5.0, 5.0),
        x0=-0.5,
        width=0.4,
        c=0.2
    )
    L.export_to_jsonfile(f"{testdir}/resources/test_export_load_file.json")

    L2 = LorentzianLine.load_from_jsonfile(f"{testdir}/resources/test_export_load_file.json")
    assert L.name == L2.name

    L3 = LorentzianLine.load_from_dict(**L2.export_to_dict())

#------------------------------------------------------------------------------

if __name__ == "__main__":
    test_Lorentzian_normalization()