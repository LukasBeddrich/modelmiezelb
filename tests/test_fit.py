import matplotlib.pyplot as plt
import os
###############################################################################
from numpy import array, linspace, logspace, tile, trapz, all, isclose, abs, random
from pprint import pprint
###############################################################################
from modelmiezelb.correction import DetectorEfficiencyCorrectionFactor
from modelmiezelb.lineshape import LorentzianLine
from modelmiezelb.sqe_model import SqE
from modelmiezelb.transformer import SqtTransformer
from modelmiezelb.fit import FitModelCreator
###############################################################################
from modelmiezelb.utils.util import MIEZE_DeltaFreq_from_time, energy_from_lambda, MIEZE_phase, detector_efficiency, triangle_distribution
from modelmiezelb.utils.helpers import flatten_list
###############################################################################
# Path quarrels
testdir = os.path.dirname(os.path.abspath(__file__))

def test_from_sqe():
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=2)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-1.0, width=0.4, c=0.0, weight=1)
    L1g = LorentzianLine("Lorentzian1g", (-5.0, 5.0), x0=0.0, width=0.5, c=0.0, weight=1)
    L2g = LorentzianLine(name="Lorentzian2g", domain=(-5.0, 5.0), x0=-1.5, width=0.3, c=0.0, weight=1)
    # Contruct a SqE model
    sqe1 = SqE((L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)
    sqe1g = SqE((L1g, L2g), lam=6.0, dlam=0.12, lSD=3.43, T=20)

    # Create some data and a Minuit obj from sqe
    x = linspace(-5,5,26)
    y = sqe1(x) * 1000
    yerr = y**0.5
    y += random.randn(len(x)) * yerr
    m = FitModelCreator.from_sqe(sqe1, x, y/1000, yerr/1000, [0.5, 0.0, 1.7, -1.5, 0.3, 0.0, 1])

    fmin, res = m.migrad()
   # print(fmin)
   # print(res)

    dL1res = {
        "x0" : 0.0,
        "width" : res[0].value,
        "c" : res[1].value,
        "weight" : res[2].value
    }
    dL2res = {
        "x0" : res[3].value,
        "width" : res[4].value,
        "c" : res[5].value,
        "weight" : res[6].value
    }

    L1res = LorentzianLine(name="Lorentzian1res", domain=(-5.0, 5.0), **dL1res)
    L2res = LorentzianLine(name="Lorentzian2res", domain=(-5.0, 5.0), **dL2res)
    sqe1res = SqE((L1res, L2res), lam=6.0, dlam=0.12, lSD=3.43, T=20)

    # # visualize
    # e = linspace(-5,5,151)
    # plt.errorbar(x, y/1000, yerr/1000, ls="", marker="o")
    # plt.plot(e, sqe1(e), color="C0", label="original")
    # plt.plot(e, sqe1g(e), color="C1", ls="--", label="initial")
    # plt.plot(e, sqe1res(e), color="C3", label="fit-res")
    # plt.legend()
    # plt.show()

#------------------------------------------------------------------------------

def test_from_transformer():
    ### Creating a SqE model for transformation
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=1)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=1.5, width=0.4, c=0.0, weight=2)
    # Contruct a SqE model
    sqe1 = SqE((L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)

    ### Instantiate a transformer
    sqt1 = SqtTransformer(
        sqe1,
        corrections=(), #(DetectorEfficiencyCorrectionFactor(sqe1, ne=15000, nlam=20),),
        ne=10000,
        nlam=20,
        lSD=3.43
    )

    ### Values for transformation
    datataus = array([2.0e-6, 6.0e-6, 4.1e-5, 2.5e-4, 5.5e-4, 1.0e-3, 1.32e-3, 1.7e-3, 2.2e-3, 2.63e-3, 2.77e-3, 3.3e-3, 4.27e-3, 5.11e-3, 6.77e-3, 8.96e-3, 2.0e-2])
    x = MIEZE_DeltaFreq_from_time(datataus*1.0e-9, 3.43, 6.0)

    ### create artificial data via TRANSFORM!!
    y = array([sqt1(freq) for freq in x])
    yerr = 0.02 * random.randn(len(y))

    ### instntiate Minuit obj!
    m = FitModelCreator.from_transformer(sqt1, x, y+yerr, yerr, [0.45, 0.0, 1.2, -1.5, 0.3, 0.0, 2.2])

#------------------------------------------------------------------------------

def test_from_Minuit_and_adjust_it():
    ### Creating a SqE model for transformation
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=1)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=1.5, width=0.4, c=0.0, weight=2)
    # Contruct a SqE model
    sqe1 = SqE((L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)

    ### Instantiate a transformer
    sqt1 = SqtTransformer(
        sqe1,
        corrections=(), #(DetectorEfficiencyCorrectionFactor(sqe1, ne=15000, nlam=20),),
        ne=10000,
        nlam=20,
        lSD=3.43
    )

    ### Values for transformation
    datataus = array([2.0e-6, 6.0e-6, 4.1e-5, 2.5e-4, 5.5e-4, 1.0e-3, 1.32e-3, 1.7e-3, 2.2e-3, 2.63e-3, 2.77e-3, 3.3e-3, 4.27e-3, 5.11e-3, 6.77e-3, 8.96e-3, 2.0e-2])
    x = MIEZE_DeltaFreq_from_time(datataus*1.0e-9, 3.43, 6.0)

    ### create artificial data via TRANSFORM!!
    y = array([sqt1(freq) for freq in x])
    yerr = 0.02 * random.randn(len(y))

    ### initialize Minuits!
    m = FitModelCreator.from_transformer(sqt1, x, y+yerr, yerr, [0.45, 0.0, 1.2, -1.5, 0.3, 0.0, 2.2])
    m.fixed["c_Lorentzian1"] = True
    m.fixed["c_Lorentzian2"] = True

    # Adjust fit behavior -> reinitialize another Minuit obj
    m2 = FitModelCreator.from_Minuit(
        minuitobj=m,
        limit_weight_Lorentzian1=(0, None),
        limit_width_Lorentzian1=(0, None),
        limit_weight_Lorentzian2=(0, None),
        limit_width_Lorentzian2=(0, None),
        fix_c_Lorentzian1=True,
        fix_c_Lorentzian2=True
    )

#------------------------------------------------------------------------------

def slow_vis_fitting():
    ### Creating a SqE model for transformation
    # We need some lines
    L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=1)
    L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=1.5, width=0.4, c=0.0, weight=2)
    # Contruct a SqE model
    sqe1 = SqE((L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)

    ### Instantiate a transformer
    # Consider detector efficiency
    decf = DetectorEfficiencyCorrectionFactor(sqe1)
    sqt1 = SqtTransformer(
        sqe1,
        corrections=(decf,),
        ne=10000,
        nlam=20,
        lSD=3.43
    )

    ### Creating a logger for debugging:
    import logging
    # Create and configure logger
    logging.basicConfig(filename=f"{testdir}/resources/fit.log", level=logging.INFO, filemode="w")
    logger = logging.getLogger()

    ### Values for transformation
    taus = logspace(-6, -1, 101)
    datataus = array([2.0e-6, 6.0e-6, 4.1e-5, 2.5e-4, 5.5e-4, 1.0e-3, 1.32e-3, 1.7e-3, 2.2e-3, 2.63e-3, 2.77e-3, 3.3e-3, 4.27e-3, 5.11e-3, 6.77e-3, 8.96e-3, 2.0e-2])
    x = MIEZE_DeltaFreq_from_time(datataus*1.0e-9, 3.43, 6.0)
    longx = MIEZE_DeltaFreq_from_time(taus*1.0e-9, 3.43, 6.0)

    ### TRANSFORM!!
    y = array([sqt1(freq) for freq in x])
    sqt1vals = array([sqt1(freq) for freq in longx])
    yerr = 0.05 * random.randn(len(y))

    ### FIT!
    m = FitModelCreator.from_transformer(sqt1, x, abs(y+yerr), yerr, [0.6, 0.0, 1.0, -1.5, 0.2, 0.0, 3.0], logger=logger)
    m.fixed["c_Lorentzian1"] = True
    m.fixed["c_Lorentzian2"] = True
    print("fcn for m:\n", m.fcn)

    m2 = FitModelCreator.from_Minuit(
        minuitobj=m,
        limit_weight_Lorentzian1=(0, None),
        limit_width_Lorentzian1=(0, None),
        limit_weight_Lorentzian2=(0, None),
        limit_width_Lorentzian2=(0, None),
        fix_c_Lorentzian1=True,
        fix_c_Lorentzian2=True,
        fix_weight_Lorentzian1=True
    )
    print("fcn for m2:\n", m2.fcn)
    
    fmin, res = m.migrad(ncall=1000)
    print(fmin)
    print(res)
#    fmin2, res2 = m2.migrad(ncall=1000)
#    print(fmin2)
#    print(res2)

    ### Gather Fit results
    dL1res1 = {
        "x0" : 0.0,
        "width" : res[0].value,
        "c" : res[1].value,
        "weight" : res[2].value
    }
    dL2res1 = {
        "x0" : res[3].value,
        "width" : res[4].value,
        "c" : res[5].value,
        "weight" : res[6].value
    }

    L1.update_line_params(**dL1res1)
    L2.update_line_params(**dL2res1)
    sqtres1vals = array([sqt1(freq) for freq in longx])

    # ### Gather Fit results
    # dL1res2 = {
    #     "x0" : 0.0,
    #     "width" : res2[0].value,
    #     "c" : res2[1].value,
    #     "weight" : res2[2].value
    # }
    # dL2res2 = {
    #     "x0" : res2[3].value,
    #     "width" : res2[4].value,
    #     "c" : res2[5].value,
    #     "weight" : res2[6].value
    # }

    # L1.update_line_params(**dL1res2)
    # L2.update_line_params(**dL2res2)
    # sqtres2vals = array([sqt1(freq) for freq in longx])

    ### Calculate init guess
    dL1ig = {
        "x0" : 0.0,
        "width" : 0.6,
        "c" : 0,
        "weight" : 1.0
    }
    dL2ig = {
        "x0" : -1.5,
        "width" : 0.2,
        "c" : 0.0,
        "weight" : 3.0
    }

    L1.update_line_params(**dL1ig)
    L2.update_line_params(**dL2ig)
    sqtigvals = array([sqt1(freq) for freq in longx])

    plt.plot(taus, sqt1vals, label="orig. curve", ls="--", color="C0")
    plt.plot(taus, sqtres1vals, label="fit curve 1", ls="-", color="C1")
    # plt.plot(taus, sqtres2vals, label="fit curve 2", ls=":", color="C4")
    plt.plot(taus, sqtigvals, label="init guess", ls="-.", color="C2")
    plt.errorbar(datataus, abs(y+yerr), yerr, label="data", ls="", marker="o", color="C0")
    plt.xscale("log")
    plt.xlabel("$\\tau_{MIEZE}$ [ns]", fontsize=16.)
    plt.ylabel("S(q,t) [arb. u.]", fontsize=16.)
    plt.legend()
    plt.show()

if __name__ == "__main__":
#    test_LeastSquareCollector()
#    test_from_sqe()
#    test_from_transformer()
    slow_vis_fitting()