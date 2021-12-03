import os
import matplotlib.pyplot as plt
import numpy as np
###############################################################################
from modelmiezelb.sqe_model import SqE, SqE_kf, SqE_from_arg, UPPER_INTEGRATION_LIMIT
from modelmiezelb.lineshape import Line, LorentzianLine, F_ILine
from modelmiezelb.utils.util import energy_from_lambda, from_energy_to_lambdaf
###############################################################################
from pprint import pprint
###############################################################################
# Path quarrels
testdir = os.path.dirname(os.path.abspath(__file__))

def we_need_some_Lines():
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=2)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-1.0, width=0.1, c=0.02, weight=1)
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
    sqecombined = SqE_kf(lines=(L3, L4), lam=6.0, dlam=0.12, lSD=3.43, T=628.0)
    valdom, errdom = quad(sqecombined, *new_domain)
    print(f"Integration value over domain from {new_domain[0]:.5f} to {UPPER_INTEGRATION_LIMIT}: {valdom:.5f} +- {errdom:.5f}")#"   |   normalization factor: {n:.5f}")
    sqecombined.update_domain((-5, 5))
    valover, errover = quad(sqecombined, -2, 2)
    print(f"Integration value in sub-domain from -2.0 to 2.0: {valover:.5f} +- {errover:.5f}")

#------------------------------------------------------------------------------

def test_sqe_arg():
    sqe_arg = SqE_from_arg(T=20.)

#------------------------------------------------------------------------------

def test_export_load():
    # We need some lines
    L1, L2 = we_need_some_Lines()
    L3 =   F_ILine("F_I1", (-energy_from_lambda(6.0), 15), x0=-0.1, width=0.008, A=350.0, q=0.02, kappa=0.01, c=0.0, weight=1)
    # Contruct a SqE model
    sqe = SqE_kf(lines=(L1, L2, L3), lam=6.0, dlam=0.12, lSD=3.43, T=20)
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
    sqe = SqE_kf(lines=(L1, L2, L3), lam=6.0, dlam=0.12, lSD=3.43, T=20)
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
    sqe = SqE_kf(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    print("L1 peak domain: ", L1.get_peak_domain())
    print("L2 peak domain: ", L2.get_peak_domain())
    print("sqe peak domain: ", sqe.get_peak_domain())

#------------------------------------------------------------------------------

def test_get_adaptive_integration_grid():
    # We need some lines
    L1, L2 = we_need_some_Lines()
    L3 = F_ILine("FI1", (-energy_from_lambda(6.0), 15), x0=-0.1, width=0.008, A=350.0, q=0.02, kappa=0.01, c=0.0, weight=1)

    # Contruct a SqE model
    sqe = SqE_kf(lines=(L1, L2, L3), lam=6.0, dlam=0.12, lSD=3.43, T=20)


    print(sqe.quasielastic.get_adaptive_integration_grid(L1, 10))
    print(sqe.inelastic.get_adaptive_integration_grid(L2, 5))
    print(sqe.get_adaptive_integration_grid(ne=5, nlam=3).shape)

    # # visualize
    # plt.hist(sqe.get_adaptive_integration_grid(ne=50, nlam=1)[0], bins=20, range=(-3.0, 3))
    # plt.plot(sqe.get_adaptive_integration_grid(ne=50, nlam=1)[0], sqe(sqe.get_adaptive_integration_grid(ne=50, nlam=1)[0]), color="limegreen")
    # plt.show()

#------------------------------------------------------------------------------

def test_SqEkf_normalization():
    lam0 = 6.0
    # We need some lines
    L1, L2 = we_need_some_Lines()
    L1.line_params["weight"] = 0.01
    L1.line_params["width"] = 0.01
    L2.line_params["weight"] = 3.0
    L2.line_params["width"] = 0.04
    L2.line_params["x0"] = 0.048
    L2.line_params["c"] = 0.0

    # Construct a SqE model
    new_domain = (-0.99999 * energy_from_lambda(lam0), UPPER_INTEGRATION_LIMIT)
    sqe = SqE_kf(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=300.0)
    sqe.update_domain(new_domain)

    # Building ee and ll arrays for caluclations
    l = np.linspace(1-0.12, 1+0.12, 5) * 6.0
    a = -0.99999 * energy_from_lambda(l)

    ee = sqe.get_adaptive_integration_grid(500, 5)

    ee = np.where(ee <= np.atleast_2d(a), np.atleast_2d(a), ee)

    ne = ee.shape[0]

    ll = np.tile(l, (ne, 1))

    print(ee.shape, ll.shape)
    print(ee[::150])
    print(ll[::150])

    sqekfvals = sqe(ee, ll)
    print(sqekfvals[0])

    # i = 0
    # while i < 5:
    #     plt.plot(ee[:,i], sqekfvals[:,i])
    #     i += 1
    # plt.show()

#------------------------------------------------------------------------------

def active_development():
    lam0 = 6.0
    # We need some lines
    L1, L2 = we_need_some_Lines()
    L1.line_params["weight"] = 0.01
    L1.line_params["width"] = 0.01
    L2.line_params["weight"] = 3.0
    L2.line_params["width"] = 0.04
    L2.line_params["x0"] = 0.048
    L2.line_params["c"] = 0.0

    # Construct a SqE model
    new_domain = (-0.99999 * energy_from_lambda(lam0), UPPER_INTEGRATION_LIMIT)
    # new_domain = (-0.99999 * energy_from_lambda(lam0), 200)
    sqe = SqE(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=300.0)
    sqekf = SqE_kf(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=300.0)
    sqe.update_domain(new_domain)
    sqekf.update_domain(new_domain)

    # # Compare SqE und kf*SqE visually
    # des = np.linspace(*new_domain, 100001)
    # sqevals = sqe(des)
    # sqekfvals = sqekf(des)
    # sqe_timekf_vals = sqe(des) * 2 * np.pi / from_energy_to_lambdaf(des, lam0)

    # plt.plot(des, sqevals, color="C0")
    # plt.plot(des, sqe_timekf_vals, ls="--", color="C3")
    # plt.plot(des, sqekfvals, ls="-.", color="C2")
    # # plt.plot(des, np.gradient(sqe_timekf_vals), ls="-.", color="limegreen")
    # plt.yscale("log")
    # plt.show()

    # Check integration over energy transfer
    from scipy.integrate import quad
    sqevaldom, sqeerrdom = quad(sqe, *new_domain)
    print(f"Integration of SqE over domain from {new_domain[0]:.5f} to {new_domain[1]}: {sqevaldom:.5f} +- {sqeerrdom:.5f}")
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    sqe_timeskf = lambda x, lam0: sqe(x) * 2 * np.pi / from_energy_to_lambdaf(x, lam0)
    sqekfvaldom, sqekferrdom = quad(sqe_timeskf, *new_domain, args=(lam0,))
    print(f"Integration of SqE*kf over domain from {new_domain[0]:.5f} to {new_domain[1]}: {sqekfvaldom:.5f} +- {sqekferrdom}")
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    kf_func = lambda x, lam0: 2 * np.pi / from_energy_to_lambdaf(x, lam0)
    kfvaldom, kferrdom = quad(kf_func, *new_domain, args=(lam0,))
    print(f"Integration of kf over domain from {new_domain[0]:.5f} to {new_domain[1]}: {kfvaldom:.5f} +- {kferrdom:.5f}")
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print(f"Subtract kf integration value from Sqe*kf value {new_domain[0]:.5f} to {new_domain[1]}: {sqekfvaldom - kfvaldom:.5f}")
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    sqe_kfvaldom, sqe_kferrdom = quad(sqekf, *new_domain)
    print(f"Integration of SqE_kf over domain from {new_domain[0]:.5f} to {new_domain[1]}: {sqe_kfvaldom:.5f} +- {sqe_kferrdom}")
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print(f"")


#------------------------------------------------------------------------------

if __name__ == "__main__":
#    test_sqe_normalization()
#    test_sqe_arg()
#    test_export_load()
#    test_update_params()
#    test_get_peak_domain()
#    test_get_adaptive_integration_grid()
    test_SqEkf_normalization()
    # active_development()