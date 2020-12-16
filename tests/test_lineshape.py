import numpy as np
import pytest
import matplotlib.pyplot as plt
import os
#------------------------------------------------------------------------------
from modelmiezelb.utils.lineshape_functions import gaussian, lorentzian, fqe_I
from modelmiezelb.utils.util import energy_from_lambda
from modelmiezelb.lineshape import Line, LorentzianLine, F_ILine, F_cLine, LineFactory, QuasielasticCalcStrategy, InelasticCalcStrategy
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

    l3 = F_cLine("FcLine1", (-15, 15), x0=0.0, width=0.02, A=350.0, q=0.023, c=0.0, weight=1.0)
    l3.update_line_params(width=0.1)
    l3.update_domain((-1.0, 1.0))

    l4 = F_ILine("FILine1", (-15, 15), x0=0.0, width=0.02, A=350.0, q=0.023, kappa=0.01, c=0.0, weight=1.0)
    l4.update_line_params(kappa=0.015)

    e = np.linspace(-1.5, 1.5, 16)
    print("Returned by 'l1.calc': ", l1.calc(e, **l1.line_params))
    print("Returned by 'l2(e)': ", l2(e))
    print("Returned by 'F_cLine': ", l3(e))
    print("Returned by 'F_ILine': ", l4(e))

#------------------------------------------------------------------------------

def test_normalization():
    Lorentzian = LorentzianLine("Lorentzian1", (-5., 5), x0=0.0, width=0.5, c=0.0)
    n = Lorentzian.normalize()

    from scipy.integrate import quad
    to_integrate = lambda x: Lorentzian(x)
    val, err = quad(to_integrate, min(Lorentzian.domain), max(Lorentzian.domain))
    print(f"Integration value: {val:.5f} +- {err:.5f}   |   normalization factor: {n:.5f}")
    print(f"Normalized Line area: {val*n}")

    # - - - - - - - - - - - - - - - - - - - -

    F_c = F_cLine("FcLine1", (-15, 15), x0=0.0, width=0.02, A=350.0, q=0.023, c=0.0, weight=1.0)
    nfc = F_c.normalize()
    ifc = F_c.integrate()

    x = np.linspace(-1000, 1000, 5000000)
    y = F_c.calc(x, **F_c.line_params)
    ifctrapz = np.trapz(y, x)
    ifcquadv, ifcquade = quad(lambda x: nfc * F_c(x), min(F_c.domain), max(F_c.domain))
    print(f"{F_c.line_params}")
    print(f"Standard Integration value for 10000 steps: {ifc}")
    print(f"QUADPACK Integration value (after normal.): {ifcquadv} +- {ifcquade}")
    print(f"TRAPEZOID Integration value -1e3 from 1e3 : {ifctrapz}")

    # - - - - - - - - - - - - - - - - - - - -

    F_I = F_ILine("FILine1", (-15, 15), x0=0.0, width=0.02, A=350.0, q=0.023, kappa=0.01, c=0.0, weight=1.0)
    nfI = F_I.normalize()
    ifI = F_I.integrate()

    x = np.linspace(-1000, 1000, 5000000)
    y = F_I.calc(x, **F_I.line_params)
    ifItrapz = np.trapz(y, x)
    ifIquadv, ifIquade = quad(lambda x: F_I(x), min(F_I.domain), max(F_I.domain))
    print(f"{F_I.line_params}")
    print(f"Standard Integration value for 10000 steps: {ifI}")
    print(f"QUADPACK Integration value                : {ifIquadv} +- {ifIquade}")
    print(f"TRAPEZOID Integration value -1e3 from 1e3 : {ifItrapz}")

#------------------------------------------------------------------------------

def test_domainenforcement():
    Lorentzian = LorentzianLine("Lorentzian1", (-5., 5), x0=-0.3, width=0.5, c=0.2)
    F_c = F_cLine("FcLine1", (-15, 15), x0=0.0, width=0.02, A=350.0, q=0.023, c=0.0, weight=1.0)

    e_in_domain = np.linspace(-5.0, 5.0, 11)
    e_beyond_domain = np.linspace(-10.0, 10.0, 21)

    pL1 = Lorentzian(e_in_domain)
    pL2 = Lorentzian(e_beyond_domain)

    pF1 = F_c(e_in_domain)
    pF2 = F_c(e_beyond_domain)

    assert np.all(pL1 == pL2[Lorentzian.within_domain(e_beyond_domain)])
    assert np.all(pF1 == pF2[Lorentzian.within_domain(e_beyond_domain)])

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

    F_c = LineFactory.create(
        "f_c",
        name="FcLine1",
        domain=(-15, 15),
        x0=0.0, width=0.02,
        A=350.0,
        q=0.023,
        c=0.0,
        weight=1.0
    )
    assert isinstance(L, LorentzianLine)
    assert isinstance(F_c, Line)

    F_I = LineFactory.create(
        "f_I",
        name="FILine1",
        domain=(-15, 15),
        x0=0.0, width=0.02,
        A=350.0,
        q=0.023,
        c=0.0,
        weight=1.0,
        kappa=0.01
    )
    assert isinstance(L, LorentzianLine)
    assert isinstance(F_c, Line)
    assert not isinstance(F_I, F_cLine)

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

    F_c1 = LineFactory.create(
        "f_c",
        name="FcLine1",
        domain=(-15, 15),
        x0=0.0, width=0.02,
        A=350.0,
        q=0.023,
        c=0.0,
        weight=1.0
    )
    F_c1.export_to_jsonfile(f"{testdir}/resources/test_export_load_F_c_file.json")
    F_c2 = F_cLine.load_from_jsonfile(f"{testdir}/resources/test_export_load_F_c_file.json")
    assert F_c1.line_params["width"] == F_c2.line_params["width"]
    F_c3 = F_cLine.load_from_dict(**F_c2.export_to_dict())
    assert min(F_c3.domain) == min(F_c1.domain)

    F_I1 = LineFactory.create(
        "f_I",
        name="FILine1",
        domain=(-15, 15),
        x0=0.0, width=0.02,
        A=350.0,
        q=0.023,
        c=0.0,
        weight=1.0,
        kappa=0.01
    )
    F_I1.export_to_jsonfile(f"{testdir}/resources/test_export_load_F_I_file.json")
    F_I2 = F_ILine.load_from_jsonfile(f"{testdir}/resources/test_export_load_F_I_file.json")
    assert F_I1.line_params["width"] == F_I2.line_params["width"]
    F_I3 = F_ILine.load_from_dict(**F_I2.export_to_dict())
    assert min(F_I3.domain) == min(F_I1.domain)

#------------------------------------------------------------------------------

def test_get_param_names():
    L1 = LorentzianLine(
        name="Lorentzian",
        domain=(-5.0, 5.0),
        x0=-0.5,
        width=0.4,
        c=0.2
    )
    F_c1 = F_cLine(
        name="Fc1",
        domain=(-5.0, 5.0),
        x0=-0.0,
        width=0.4,
        c=0.2
    )
    print(L1.get_param_names())
    print(F_c1.get_param_names())

#------------------------------------------------------------------------------

def test_update_line_params():
    L = LorentzianLine(
        name="Lorentzian",
        domain=(-5.0, 5.0),
        x0=-0.5,
        width=0.4,
        c=0.2
    )
    print(L.export_to_dict())
    tdict = dict(x0=1.0, c=0.0)
    L.update_line_params(**tdict)
    print("After update: ", L.export_to_dict())

#------------------------------------------------------------------------------

def test_get_peak_domain():
    L = LorentzianLine(
        name="Lorentzian",
        domain=(-15.0, 15.0),
        x0=-0.0,
        width=0.4,
        c=0.2
    )
    print(f"The peak domain: {L.get_peak_domain()}")

    etotal = np.linspace(*L.domain, num=10000)
    epeak = np.linspace(*L.get_peak_domain(), num=1333)
    e = np.concatenate((
        np.linspace(L.domain[0], L.get_peak_domain()[0], 500),
        epeak,
        np.linspace(L.get_peak_domain()[1], L.domain[1], 500)
    ))

    print(f"Integral (trapz) over total domain: {np.trapz(L(etotal), etotal)}")
    print(f"Integral (trapz) over total (splitted) domain: {np.trapz(L(e), e)}")
    print(f"Integral (trapz) over peak domain: {np.trapz(L(epeak), epeak)}")

#------------------------------------------------------------------------------

def test_get_peak_domain_strategy():
    inelstrat = InelasticCalcStrategy(20.0)
    Lwide = LorentzianLine(
        name="Lwide",
        domain=(-15.0, 15.0),
        x0=-0.5,
        width=0.4,
        c=0.0
    )
    Lnarrow = LorentzianLine(
        name="Lnarrow",
        domain=(-15.0, 15.0),
        x0=-0.5,
        width=0.07,
        c=0.0
    )
    print(inelstrat.get_peak_domain(Lwide))
    print(inelstrat.get_peak_domain(Lnarrow))

#------------------------------------------------------------------------------

# def test_get_adaptive_integration_grid():
#     quasistrat = QuasielasticCalcStrategy()
#     inelstrat = InelasticCalcStrategy(20.0)
#     Lwide = LorentzianLine(
#         name="Lwide",
#         domain=(-15.0, 15.0),
#         x0=-0.5,
#         width=0.4,
#         c=0.0
#     )
#     Lnarrow = LorentzianLine(
#         name="Lnarrow",
#         domain=(-15.0, 15.0),
#         x0=-0.5,
#         width=0.07,
#         c=0.0
#     )
#     Lquasi = LorentzianLine(
#         name="Lquasi",
#         domain=(-15.0, 15.0),
#         x0=0.0,
#         width=0.1,
#         c=0.0
#     )

#     print("Quasielastic domain retrieval.")
#     print(quasistrat.get_peak_domain(Lquasi))
#     print(quasistrat.get_adaptive_integration_grid(Lquasi, 5))

#     print("Inelelastic domain retrieval.")
#     print(inelstrat.get_peak_domain(Lnarrow))
#     print(inelstrat.get_adaptive_integration_grid(Lnarrow, 5))
#     print(inelstrat.get_peak_domain(Lwide))
#     print(inelstrat.get_adaptive_integration_grid(Lwide, 5))

#------------------------------------------------------------------------------

def visualize_Lines():

    Lorentzian = LorentzianLine("Lorentzian1", (-energy_from_lambda(6.0), 15), x0=-0.3, width=0.5, c=0.0)
    F_c = F_cLine("F_c1", (-energy_from_lambda(6.0), 15), x0=-0.3, width=0.5, A=350.0, q=0.02, c=0.0, weight=1)
    F_I = F_ILine("F_I1", (-energy_from_lambda(6.0), 15), x0=-0.3, width=0.008, A=350.0, q=0.02, kappa=0.01, c=0.0, weight=1)
    print(F_I.integrate())
    F_I2 = F_ILine("F_I2", (-energy_from_lambda(6.0), 15), x0=-0.3, width=0.1, A=367.0, q=0.124, kappa=0.065, c=0.0, weight=1)
    print(F_I2.integrate())

    e = np.linspace(-0.5, 0.0, 1201)

    plt.plot(e, Lorentzian(e))
    plt.plot(e, F_c(e) * F_c.normalize(), ls="--", lw=2.0)
    plt.plot(e, F_I(e), ls="dotted", lw=4.0)
    plt.plot(e, F_I2(e), ls="-.", lw=1.0)
    plt.show()

#------------------------------------------------------------------------------

def test_get_adaptive_integration_grid():
    ### Lines
    LL1 = LorentzianLine("LL1", (-energy_from_lambda(6.0), 15), x0=0.048, width=0.04, c=0.0, weight=0.0)
    F_c = F_cLine("F_c1", (-energy_from_lambda(6.0), 15), x0=0.0, width=0.02, A=350.0, q=0.02, c=0.0, weight=1)
    F_I = F_ILine("F_I1", (-energy_from_lambda(6.0), 15), x0=-0.02, width=0.008, A=350.0, q=0.02, kappa=0.01, c=0.0, weight=1)

    ### CalcStrategy
    inel = InelasticCalcStrategy(610)
    quasi = QuasielasticCalcStrategy()

    # ### Vis
    # ne = 500
    # plt.plot(LL1.get_adaptive_integration_grid(ne), [0] * (ne * LL1.INTEGRATION_POINT_NUMBER_FACTOR - 1), ls="", color="C0", marker=".", mec="None", label=f"Line {LL1.name}")
    # plt.plot(F_c.get_adaptive_integration_grid(ne), [1] * (ne * F_c.INTEGRATION_POINT_NUMBER_FACTOR - 1), ls="", color="C1", marker=".", mec="None", label=f"Line {F_c.name}")
    # plt.plot(F_I.get_adaptive_integration_grid(ne), [2] * (ne * F_I.INTEGRATION_POINT_NUMBER_FACTOR - 1), ls="", color="C2", marker=".", mec="None", label=f"Line {F_I.name}")
    # plt.plot(inel.get_adaptive_integration_grid(LL1, ne), [0.2] * (ne * LL1.INTEGRATION_POINT_NUMBER_FACTOR - 2), ls="", color="C0", marker="2", label=f"Calcstrat {LL1.name}")
    # plt.plot(quasi.get_adaptive_integration_grid(F_c, ne), [1.2] * (ne * F_c.INTEGRATION_POINT_NUMBER_FACTOR - 2), ls="", color="C1", marker="2", label=f"Calcstrat {F_c.name}")
    # plt.plot(inel.get_adaptive_integration_grid(F_I, ne), [2.2] * (ne * F_I.INTEGRATION_POINT_NUMBER_FACTOR - 2), ls="", color="C2", marker="2", label=f"Calcstrat {F_I.name}")
    # plt.legend()
    # plt.figure()
    # plt.hist(inel.get_adaptive_integration_grid(LL1, ne), range=(-0.1, 0.1), bins=20, label="LL1")
    # plt.legend()
    # plt.figure()
    # plt.hist(inel.get_adaptive_integration_grid(F_c, ne), range=(-0.1, 0.1), bins=20, label="F_c")
    # plt.legend()
    # plt.figure()
    # plt.hist(inel.get_adaptive_integration_grid(F_I, ne), range=(-0.1, 0.1), bins=20, label="F_I")
    # plt.legend()
    # plt.show()

#------------------------------------------------------------------------------

if __name__ == "__main__":
#    test_Lines()
#    test_normalization()
#    test_export_load()
#    test_get_param_names()
#    test_update_line_params()
#    test_get_peak_domain()
#    test_get_peak_domain_strategy()
    test_get_adaptive_integration_grid()
#    visualize_Lines()
    