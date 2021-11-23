import matplotlib.pyplot as plt
import os
###############################################################################
from numpy import linspace, logspace, tile, trapz, all, isclose, abs, array
from pprint import pprint
###############################################################################
from modelmiezelb.correction import CorrectionFactor, DetectorEfficiencyCorrectionFactor, EnergyCutOffCorrectionFactor
from modelmiezelb.lineshape import LorentzianLine, F_ILine, F_cLine
from modelmiezelb.sqe_model import SqE, SqE_from_arg
from modelmiezelb.transformer import SqtTransformer, STANDARD_SQT
###############################################################################
from modelmiezelb.utils.util import MIEZE_DeltaFreq_from_time, energy_from_lambda, MIEZE_phase, detector_efficiency, triangle_distribution
###############################################################################
import modelmiezelb.arg_inel_mieze_model as arg
###############################################################################
# Path quarrels
testdir = os.path.dirname(os.path.abspath(__file__))

def test_transformer_init():
    ### Creating a SqE model for transformation
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=2)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-1.0, width=0.4, c=0.0, weight=1)
    L3 =   F_ILine("FI1", (-energy_from_lambda(6.0), 15), x0=-0.1, width=0.008, A=350.0, q=0.02, kappa=0.01, c=0.0, weight=1)
    # Contruct a SqE model
    sqe1 = SqE(lines=(L2,), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    sqe2 = SqE(lines=(L1, L2, L3), lam=6.0, dlam=0.12, lSD=3.43, T=20)

    ### Instantiate a transformer
    SqtTransformer(sqe2, nlam=20, ne=10000)

#------------------------------------------------------------------------------

def test_transformer_basics():
    ### Creating a SqE model for transformation
    # We need some lines
    L1 = LorentzianLine(name="Lorentzian1", domain=(-15.0, 15.0), x0=-1.0, width=0.4, c=0.0, weight=3)
    L3 =   F_ILine("FI1", (-energy_from_lambda(6.0), 15), x0=-0.1, width=0.008, A=350.0, q=0.02, kappa=0.01, c=0.0, weight=1)
    # Contruct a SqE model
    sqe1 = SqE(lines=(L1, L3), lam=6.0, dlam=0.12, lSD=3.43, T=20)

    ### Instantiate a transformer
    sqt1 = SqtTransformer(
        sqe1,
        corrections=(DetectorEfficiencyCorrectionFactor(sqe1, ne=15000, nlam=20),),
        ne=15000,
        nlam=20,
        lSD=3.43
    )

    ### Values for transformation
    taus = logspace(-6, -1, 11)
    freqs = MIEZE_DeltaFreq_from_time(taus*1.0e-9, 3.43, 6.0)

    ### TRANSFORM!!
    sqt1vals = [sqt1(freq) for freq in freqs]
    sqt1vals_arg = arg.Sqt(
        arg.fqe_I,
        freqs,
        -1.0,
        0.4,
        15000,
        20,
        3.43,
        6.0,
        0.12,
        0.0,
        1.0,
        20,
        0.00005,
        0.1,
        350.
    )

    ### Visualize results
    # print(sqt1vals)
    # plt.plot(taus, abs(sqt1vals), ls="-", marker="o", label="modelmieze")
    # plt.plot(taus, sqt1vals_arg, ls=":", marker="s", label="arg")
    # plt.xscale("log")
    # plt.legend()
    # plt.show()

#------------------------------------------------------------------------------

def test_transform_arg_model():
    ### Creating a SqE model for transformation
    sqe_arg = SqE_from_arg(T=20.)

    ### Instantiate a transformer
    sqt_arg = SqtTransformer(
        sqe_arg,
        corrections=(DetectorEfficiencyCorrectionFactor(sqe_arg, ne=15000, nlam=20),),
        ne=15000,
        nlam=20,
        lSD=3.43
    )

    ### Values for transformation
    taus = logspace(-6, -1, 11)
    freqs = MIEZE_DeltaFreq_from_time(taus*1.0e-9, 3.43, 6.0)

    ### TRANSFORM!!
    sqt_argvals = [sqt_arg(freq) for freq in freqs]
    # For the next step to work, A1 needs to be manually removed
    # from arg.Sqt calculation.
    sqt_argmodulevals = arg.Sqt(
        arg.fqe_I,
        freqs,
        -1.0,
        0.4,
        15000,
        20,
        3.43,
        6.0,
        0.12,
        0.0,
        1.0,
        20,
        0.00005,
        0.1,
        350.
    )

    # plt.plot(taus, sqt_argvals, label="arg-model")
    # plt.plot(taus, sqt_argmodulevals, label="arg-model from mod")
    # plt.legend()
    # plt.xscale("log")
    # plt.show()

#------------------------------------------------------------------------------

def test_manualtransform_arg_model():
    ### Creating a SqE model for transformation
    sqe_arg = SqE_from_arg(T=20.)
    ### Creating energy, wavelength parameter space
    lam = sqe_arg.model_params["lam"]
    dlam = sqe_arg.model_params["dlam"]
    nlam = 5   #20
    ne = 10    #15000
    lSD = 3.43

    l = linspace(1-dlam*1.01, 1+dlam*1.01, nlam) * lam
    ll = tile(l,(ne,1))
    a = -0.99999 * energy_from_lambda(l)
    ee = linspace(a, 15.0, ne)

    ### Values for transformation
    taus = logspace(-6, -1, 101)
    freqs = MIEZE_DeltaFreq_from_time(taus*1.0e-9, 3.43, 6.0)

    mieze_phase = MIEZE_phase(ee, freqs[30], lSD, ll)
    det_eff = detector_efficiency(ee, ll, 1)
    tri_distr = triangle_distribution(ll, lam, dlam)

    print(ll)
    print(ee)
    print(mieze_phase)
    print(det_eff)
    print(tri_distr)

#------------------------------------------------------------------------------

def test_export_load():
    ### Creating a SqE model for transformation
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.00005, c=0.0, weight=0.1)
    L2 =   F_ILine("FI1", (-energy_from_lambda(6.0), 15), x0=-0.01, width=0.008, A=350.0, q=0.02, kappa=0.01, c=0.0, weight=0.9)
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=628)
    # Add the detector efficiency correction
    decf = DetectorEfficiencyCorrectionFactor(sqe, ne=500, nlam=20)
    # Add the energycutoff correction
    eccf = EnergyCutOffCorrectionFactor(sqe, ne=500, nlam=20 )

    ### Instantiate a transformer
    sqt = SqtTransformer(sqe, corrections=(decf, eccf), nlam=20, ne=500, lSD=3.43, integ_mode="adaptive")

    ### Export
    sqt_dict = sqt.export_to_dict()
    pprint(sqt_dict["corrections"])
    sqt.export_to_jsonfile(f"{testdir}/resources/test_transformer_export_load_file.json")
    print("", "SqE's of the: ", f"- sqe: {sqe}", f"- decf: {decf.sqe}", f"- eccf: {eccf.sqe}", f"- sqt: {sqt.sqemodel}", sep="\n")

    ### Loading
    sqt_from_dict = sqt.load_from_dict(**sqt_dict)
    print("", "Loading successful: ", sqt, sqt_from_dict, sep='\n')
    sqt_from_file = sqt.load_from_jsonfile(f"{testdir}/resources/test_transformer_export_load_file.json")
    print("Loading successful: ", sqt, sqt_from_file, sep='\n')
    print("", "SqE's of the: ", f"- sqe: {sqe}", f"- decf: {sqt_from_dict.corrections[0].sqe}", f"- eccf: {sqt_from_dict.corrections[1].sqe}", f"- sqt: {sqt_from_dict.sqemodel}", sep="\n")

    print("\n\nThis is the sqt loaded from dict")
    pprint(sqt_from_file.export_to_dict())

#------------------------------------------------------------------------------

def test_update_params():
    ### Creating a SqE model for transformation
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=2)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-1.0, width=0.4, c=0.0, weight=1)
    L3 = F_ILine("FI1", (-energy_from_lambda(6.0), 15), x0=-0.1, width=0.008, A=350.0, q=0.02, kappa=0.01, c=0.0, weight=1)
    # Contruct a SqE model
    sqe = SqE(lines=(L1, L2, L3), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    # Add the detector efficiency correction
    decf = DetectorEfficiencyCorrectionFactor(sqe, ne=10000, nlam=20)
    # Add the energycutoff correction
    eccf = EnergyCutOffCorrectionFactor(sqe, ne=10000, nlam=20 )

    ### Instantiate a transformer
    sqt = SqtTransformer(sqe, corrections=(decf, eccf), nlam=20, ne=10000, lSD=3.43)
    print("\n\nBefore update:")
    pprint(sqt.export_to_dict())
    tdict = dict(
        T=30,
        lam=8.0,
        x0_Lorentzian2=-0.5,
        width_Lorentzian2=0.025,
        weight_Lorentzian1=5,
        nlam=55,
        kappa_FI1=1.0,
        some_wired_param=True
    )
    sqt.update_params(**tdict)
    print("\n\nAfter update:")
    pprint(sqt.export_to_dict())

#------------------------------------------------------------------------------

def test_adaptive_vs_linear():
    ### Instantiate a transformer
    sqtadapt = SqtTransformer.load_from_dict(**STANDARD_SQT.export_to_dict())
    sqtlinear = SqtTransformer.load_from_dict(**STANDARD_SQT.export_to_dict())
    sqtlinear.update_params(ne=50000, integ_mode="linear")

    ### Values for transformation
    taus = logspace(-4, 1, 81)
    freqs = MIEZE_DeltaFreq_from_time(taus*1.0e-9, 3.43, 6.0)

    ### perform transformation
    from time import time
    startt = time()
    adaptvals = array([sqtadapt(freq) for freq in freqs])
    intermedt = time()
    linearvals = array([sqtlinear(freq) for freq in freqs])
    stopt = time()

    print(f"Adaptive integration took: {intermedt - startt:.1f}")
    print(f"Linear integration took  : {stopt - intermedt:.1f}")

    # plt.plot(taus, adaptvals, marker=".", label="Adaptive")
    # plt.plot(taus, linearvals, marker=".", label="Linear")
    # plt.xscale("log")
    # plt.legend()
    # plt.show()

#------------------------------------------------------------------------------

# def test_in_development():
#     sqtadapt = SqtTransformer.load_from_dict(**STANDARD_SQT.export_to_dict())
#     sqtadapt.params.update(dict(ne=15, nlam=2))
#     print(sqtadapt(100001.0)[:,0])

#------------------------------------------------------------------------------

if __name__ == "__main__":
#    test_transformer_init()
#    test_transformer_basics()
#    test_transform_arg_model()
#    test_manualtransform_arg_model()
#    test_export_load()
    test_update_params()
#    test_adaptive_vs_linear()
#    test_in_development()