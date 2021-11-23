import matplotlib.pyplot as plt
import modelmiezelb.arg_inel_mieze_model as arg
import os
###############################################################################
from numpy import linspace, tile, trapz, all, isclose, arange, ones, atleast_2d, where
from pprint import pprint
from time import time
###############################################################################
from modelmiezelb.correction import CorrectionFactor, DetectorEfficiencyCorrectionFactor, EnergyCutOffCorrectionFactor
from modelmiezelb.lineshape import LorentzianLine, F_cLine, F_ILine
from modelmiezelb.sqe_model import SqE, UPPER_INTEGRATION_LIMIT
###############################################################################
from modelmiezelb.utils.util import detector_efficiency, triangle_distribution, energy_from_lambda, energy_lambda_nrange, cascade_efficiency
###############################################################################
from pprint import pprint
###############################################################################
# Path quarrels
testdir = os.path.dirname(os.path.abspath(__file__))

def test_CorrectionFactor_instantiation():
    corrf1 = CorrectionFactor(None)
    print(f"Instance required calculation parameters {corrf1._required_params}")
    print(f"Class required calculation parameters {CorrectionFactor._required_params}")

#------------------------------------------------------------------------------

def test_DetectorEfficiencyCorrectionFactor():
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=2)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-1.0, width=0.4, c=0.02, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    # Instantiate a detector efficiency corr factor
    decf = DetectorEfficiencyCorrectionFactor(sqe)

    ne = 10
    nlam = 5
    lam = 6.0 * linspace(1-0.12*1.01, 1+0.12*1.01, nlam)
    lams = tile(lam, (ne, 1))
    
    a = -0.99999 * energy_from_lambda(lam)
    b = 15.0 + a
    es = linspace(a, b, ne)

    deteff = detector_efficiency(es, lams, 1)
    tria = triangle_distribution(lams, 6.0, 0.12)

    print("Triangular wavelenght distr.: ", tria)
    print("Triangular wavelength distr. shape: ", tria.shape)
    print("Det. eff. values: ", deteff)
    print("Det. eff. values shape: :", deteff.shape)

    sqevals = sqe(es)

    print("Manual mult.: ", sqevals * deteff * tria)
    print("Class result: ", decf(es, lams))

    print("Are manual and deteffcorrfac identical?: ", all((sqevals * deteff * tria) == decf(es, lams)))

#------------------------------------------------------------------------------

def test_DetectorEfficiencyCorrectionFactor_compare_with_arg():
    # We need some lines
    L1 = LorentzianLine(name="Lorentzian1", domain=(-16.0, 16.0), x0=-1.0, width=0.4, c=0.0, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1,), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    # Instantiate a detector efficiency corr factor
    decf = DetectorEfficiencyCorrectionFactor(sqe)

    ne = 15000
    nlam = 20
    lam = 6.0 * linspace(1-0.12*1.01, 1+0.12*1.01, nlam)
    lams = tile(lam, (ne, 1))
    
    a = -0.99999 * energy_from_lambda(lam)
    b = 15.0 + a
    es = linspace(a, b, ne)

    test_decf_val = arg.DetFac_Eint_lamInt(
        arg.fqe_I,
        -1.0,
        0.4,
        15000,
        20,
        6.0,
        0.12,
        1,
        0.0,
        1.0,
        20,
        0.5,
        0.00001,
        350.
    )
    decf_val_man_int = decf(es, lams)
    decf_val = decf.correction(es, lams)
    print("arg res: ", test_decf_val)
    print("class calc res manualy integrated: ", decf_val_man_int)
    print("class res: ", decf_val)

#------------------------------------------------------------------------------

def test_EnergyCutOffCorrectionFactor_vs_arg():
    # We need some lines
    L1 = LorentzianLine(name="Lorentzian1", domain=(-16.0, 16.0), x0=-1.0, width=0.4, c=0.0, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1,), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    # Instantiate a energy cutoff corr factor
    eccf = EnergyCutOffCorrectionFactor(sqe)

    ne = 15000
    nlam = 20
    lam = 6.0 * linspace(1-0.12*1.01, 1+0.12*1.01, nlam)
    lams = tile(lam, (ne, 1))
    
    a = -0.99999 * energy_from_lambda(lam)
    b = 15.0 + a
    es = linspace(a, b, ne)

    test_eccf_vals = arg.CutFac_Eint(
        arg.lorentzian,
        -1.0,
        0.4,
        15000,
        20,
        6.0,
        0.12,
        1,
        0.0,
        1.0,
        20,
        0.5,
        0.00001,
        350.
    )

    eccf_vals = eccf.correction(es, lams)

#    print(test_eccf_vals.shape)
    print(test_eccf_vals)

#    print(eccf_vals.shape)  
    print(eccf_vals)
#    print(isclose(eccf_vals, test_eccf_vals, atol=0.01))
    # argsqe = arg.SvqE(arg.lorentzian,
    #     es[::50,0],
    #     -1.0,
    #     0.4,
    #     0.0,
    #     1.0,
    #     6.0,
    #     20,
    #     0.000001,
    #     0.1,
    #     350.)
#    print(argsqe)
#    plt.plot(es[::50,0], sqe(es[::50,0]))
#    plt.plot(es[::50,0], argsqe)
#    plt.show()

#------------------------------------------------------------------------------

def test_correctionFactor_dimensionality():
    # We need some lines
    L1 = LorentzianLine(name="Lorentzian1", domain=(-16.0, 16.0), x0=-1.0, width=0.4, c=0.0, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1,), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    # Instantiate a detector efficiency corr factor
    decf = DetectorEfficiencyCorrectionFactor(sqe)
    # Instantiate a energy cutoff corr factor
    eccf = EnergyCutOffCorrectionFactor(sqe)

    ne = 15
    nlam = 5
    lam = 6.0 * linspace(1-0.12*1.01, 1+0.12*1.01, nlam)
    lams = tile(lam, (ne, 1))
    
    a = -0.99999 * energy_from_lambda(lam)
    b = 15.0 + a
    es = linspace(a, b, ne)

    print(lams)
    print(es)
    print(decf.calc(es, lams))
    print(eccf.calc(es, lams))

#------------------------------------------------------------------------------

def test_EnergyCutoffCorrectionFactor():
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=2)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-1.0, width=0.4, c=0.02, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    new_domain = (-1 * energy_from_lambda(6.0), UPPER_INTEGRATION_LIMIT)
    sqe.update_domain(new_domain)
    # init energycutoff
    eccf = EnergyCutOffCorrectionFactor(sqe, ne=10000, nlam=20)

    ne = 10000
    nlam = 5
    lam = 6.0 * linspace(1-0.12*1.01, 1+0.12*1.01, nlam)
    lams = tile(lam, (ne, 1))
    
    a = -0.99999 * energy_from_lambda(lam)
    b = 15.0 + a
    es = linspace(a, b, ne)

    ### Calculate the trapz integral over the S(q,E)
    # Only over domain (interval length: 15 meV)

    I_over_dom_only = trapz(sqe(es[:, 2]), es[:, 2])
    print("Trapz integration over the domain.")
    print(f"Interval {a[2]:.4f} - {b[2]:.4f} -> length {b[2]-a[2]:.4f} meV.")
    print(f"#Steps = {ne}")
    print(f"Integral value: {I_over_dom_only:.4f}")
#    plt.plot(es[:,2], sqe(es[:,2]), label="Over domain only")
#    plt.show()

    # Beyond domain same array length
    es_same_length = linspace(-UPPER_INTEGRATION_LIMIT, UPPER_INTEGRATION_LIMIT, ne)
    I_beyond_dom_same_length = trapz(sqe(es_same_length), es_same_length)
    print("\nTrapz integration beyond the domain with varrying stepsize.")
    print(f"Interval {-UPPER_INTEGRATION_LIMIT} - {UPPER_INTEGRATION_LIMIT} -> length {30.0} meV.")
    print(f"#Steps = {ne}")
    print(f"Integral value: {I_beyond_dom_same_length:.4f}")
#    plt.plot(es_same_length, sqe(es_same_length), ls="--", label="Beyond domain ne=10000")
#    plt.show()

    # Beyond domain same step size
    es_same_stepsize = arange(-UPPER_INTEGRATION_LIMIT, UPPER_INTEGRATION_LIMIT+0.001, 15e-3)
    I_beyond_dom_same_stepsize = trapz(sqe(es_same_stepsize), es_same_stepsize)
    print("\nTrapz integration beyond the domain with varrying stepsize.")
    print(f"Interval {-UPPER_INTEGRATION_LIMIT} - {UPPER_INTEGRATION_LIMIT} -> length {30.0} meV.")
    print(f"#Steps = {30.0 / 0.015}")
    print(f"Integral value: {I_beyond_dom_same_stepsize:.4f}")
#    plt.plot(es_same_stepsize, sqe(es_same_stepsize), ls="-.", label="Beyond domain de=0.015 meV")
#    plt.show()

#------------------------------------------------------------------------------
    
def test_DetectorEfficiency_cancel():
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=2)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-1.0, width=0.4, c=0.02, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    new_domain = (-1 * energy_from_lambda(6.0), UPPER_INTEGRATION_LIMIT)
    sqe.update_domain(new_domain)
    # init energycutoff
    decf = DetectorEfficiencyCorrectionFactor(sqe, ne=10, nlam=5)

    ee, ll = energy_lambda_nrange(15.0, 6.0, 0.12, 10000, 20)

#    print(detector_efficiency(ee, ll, 1) * decf(ee, ll))
    print(trapz(trapz(decf(ee, ll) * decf.legacy_calc(ee, ll, 0), ee, axis=0), ll[0]))
#    print(trapz(trapz(ones(ll.shape), ee, axis=0), ll[0]))
    print(trapz(trapz(decf.legacy_calc(ee, ll, 1), ee, axis=0), ll[0]))
    print(trapz(trapz(decf.legacy_calc(ee, ll, 0), ee, axis=0), ll[0]))
    print(trapz(trapz(decf.legacy_calc(ee, ll, 0), ee, axis=0)/trapz(trapz(decf.legacy_calc(ee, ll, 1), ee, axis=0), ll[0]), ll[0]))
#    print(detector_efficiency(ee[::500, ::5], ll[::500, ::5], 0))
#    print()

#------------------------------------------------------------------------------

def test_DetectorEfficiency_quad_vs_trapz():
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=2)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-1.0, width=0.4, c=0.02, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)

    ### QUAD
    from scipy.integrate import dblquad
    # parameter for calculation
    plam, pdlam = 6.0, 0.12

    def dblquadfunc(energy, lam, on):
        # det_eff = detector_efficiency(energy, lam, 1)
        # sqeval = sqe(energy)
        # tri_distr_weight = triangle_distribution(lam, plam, pdlam)
        return detector_efficiency(energy, lam, on) * sqe(energy) * triangle_distribution(lam, plam, pdlam)
    # integrate
    t0quad = time()
    reson, erron = dblquad(
        dblquadfunc,
        plam * (1-pdlam),
        plam * (1+pdlam),
        lambda x: -0.9999 * energy_from_lambda(x),
        lambda x: UPPER_INTEGRATION_LIMIT,
        args=(1,),
        #epsabs=1.0e-2
    )
    resoff, erroff = dblquad(
        dblquadfunc,
        plam * (1-pdlam),
        plam * (1+pdlam),
        lambda x: -0.9999 * energy_from_lambda(x),
        lambda x: UPPER_INTEGRATION_LIMIT,
        args=(0,),
        #epsabs=1.0e-2
    )
    t1quad = time()
    print(f"RESULT: {resoff/reson} +- {resoff/reson * ((erron/reson)**2+(erroff/resoff)**2)**0.5}")
    print(f"dblquad took {t1quad - t0quad} seconds")

    ### TRAPZ
    decf = DetectorEfficiencyCorrectionFactor(sqe)
    ee, ll = energy_lambda_nrange(
        UPPER_INTEGRATION_LIMIT,
        6.0,
        0.12,
        15000,
        20
    )
    # integrate
    t0trapz = time()
    trapz_res = decf(ee, ll)
    t1trapz = time()
    print(f"RESULT: {trapz_res}")
    print(f"trapz took {t1trapz - t0trapz} seconds")

#------------------------------------------------------------------------------

def test_export_load():
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=2)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-1.0, width=0.4, c=0.02, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    new_domain = (-1 * energy_from_lambda(6.0), UPPER_INTEGRATION_LIMIT)
    sqe.update_domain(new_domain)
    # init energycutoff
    decf = DetectorEfficiencyCorrectionFactor(sqe, ne=10000, nlam=20)

    # exports
    corrdict = decf.export_to_dict()
    decf.export_to_jsonfile(f"{testdir}/resources/test_correction_export_load_file.json")

    # loading
    decf_from_dict = decf.load_from_dict(**corrdict)
    print(decf_from_dict.export_to_dict())
    print("", "Loading successful: ", decf, decf_from_dict, sep='\n')
    decf_from_jsonfile = decf.load_from_jsonfile(f"{testdir}/resources/test_correction_export_load_file.json")
    print("Loading successful: ", decf, decf_from_jsonfile, sep='\n')

#------------------------------------------------------------------------------

def test_update_params():
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=2)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-1.0, width=0.4, c=0.02, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    # init energycutoff
    decf = DetectorEfficiencyCorrectionFactor(sqe, ne=10000, nlam=20)
    pprint(decf.export_to_dict())
    tdict = dict(
        T=30,
        lam=8.0,
        x0_Lorentzian2=-0.5,
        weight_Lorentzian1=5,
        nlam=55
    )
    decf.update_params(**tdict)
    pprint(decf.export_to_dict())

#------------------------------------------------------------------------------

def test_adaptive_vs_linear():
    # 
    L1 = LorentzianLine("LL1", (-energy_from_lambda(6.0), 15), x0=0.048, width=0.04, c=0.0, weight=0.0)
    L2 = F_cLine("F_c1", (-energy_from_lambda(6.0), 15), x0=0.0, width=0.0001, A=350.0, q=0.02, c=0.0, weight=1)
    L3 = F_ILine("F_I1", (-energy_from_lambda(6.0), 15), x0=-0.02, width=0.01, A=350.0, q=0.02, kappa=0.01, c=0.0, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2, L3), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    # Add the detector efficiency correction
    decf = DetectorEfficiencyCorrectionFactor(sqe, ne=100, nlam=20)

    ### Construct the adaptive integration grid
    ne   = 100
    nlam = 21
    l = linspace(1 - sqe.model_params["dlam"], 1 + sqe.model_params["dlam"], nlam) * sqe.model_params["lam"]
    a = -0.99999 * energy_from_lambda(l)

    ee = sqe.get_adaptive_integration_grid(ne, nlam)
    ee = where(ee <= atleast_2d(a), atleast_2d(a), ee)

    ne = ee.shape[0]
    ll = tile(l,(ne, 1))

    print(f"ee: - Shape: {ee.shape}\n - ee[::50, {nlam//2}]: {ee[::50, nlam//2]}")
    print(f"ll: - Shape: {ll.shape}\n - ll[::50, {nlam//2}]: {ll[::50, nlam//2]}")

    ### Construct a standard linear integration grid
    nelin   = 10000
    nlamlin = 21
    eelin, lllin = energy_lambda_nrange(15.0, 6.0, 0.12, nelin, nlamlin)

    print(f"eelin: - Shape: {ee.shape}\n - ee[::{nelin//10}, {nlam//2}]: {eelin[::nelin//10, nlam//2]}")
    print(f"lllin: - Shape: {ll.shape}\n - ll[::{nelin//10}, {nlam//2}]: {lllin[::nelin//10, nlam//2]}")

    ### perform correction calculation
    from time import time
    startt = time()
    adaptcorr = decf.calc(ee, ll)
    intermedt = time()
    lincorr = decf.calc(eelin, lllin)
    stopt = time()

    print(f"Adaptive integration took: {intermedt - startt:.6f}")
    print(f"Adaptive grid correction value: {adaptcorr:.6f}")
    print(f"Linear integration took  : {stopt - intermedt:.6f}")
    print(f"Linear grid correction value  : {lincorr:.6f}")

#------------------------------------------------------------------------------
def test_cascade_efficiency():
    lam0 = 6.0
    des = linspace(-0.9999*energy_from_lambda(lam0), 15)
    cascade_eff_10 = cascade_efficiency(des, lam0)
    cascade_eff_6 = cascade_efficiency(des, lam0, 6)





if __name__ == "__main__":
#    test_CorrectionFactor_instantiation()
#    test_DetectorEfficiencyCorrectionFactor()
#    test_EnergyCutOffCorrectionFactor()
#    test_correctionFactor_dimensionality()
#    test_EnergyCutoffCorrectionFactor()
#    test_DetectorEfficiency_cancel()
#    test_DetectorEfficiency_quad_vs_trapz()
#    test_export_load()
#    test_update_params()
#    test_adaptive_vs_linear()
    test_cascade_efficiency()