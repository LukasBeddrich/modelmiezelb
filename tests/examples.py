### All the imports

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad

from modelmiezelb.utils.lineshape_functions import gaussian, lorentzian, fqe_I
from modelmiezelb.lineshape import Line, LorentzianLine, LineFactory, InelasticCalcStrategy, QuasielasticCalcStrategy
from modelmiezelb.sqe_model import SqE, SqE_kf
from modelmiezelb.utils.util import energy_from_lambda, wavelength_from_energy

#------------------------------------------------------------------------------
### A range in energy space for the evaluation
e = np.linspace(-5.0, 5.0, 1001)

### Create some basic Lorentzian lines
# Diese sind erstmal unabhängig von wofür sie verwendet werden
# soll heißen ob "quasielastisch" oder "inelastisch"
L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=2)
L2 = LorentzianLine("Lorentzian2", (-5.0, 5.0), x0=0.0, width=2.0, c=0.05, weight=1)
L3 = LineFactory.create('lorentzian', name="Lorentzian3", domain=(-5.0, 5.0), x0=-1.0, width=0.4, c=0.02, weight=1)

### Evaluate a Line directly
### compare it with its normalized version and plot it
f1 = plt.figure(figsize=(6.0, 4.0))
ax1 = f1.add_subplot(111)
ax1_label = f"A simple Lorentzian line\n$\int L_1(E)\,dE= ${quad(L1, *L1.domain)[0]:.4f}"
ax1.plot(e, L1(e), color="C0", lw=2.0, label=ax1_label)
ax1.plot(e, L1(e) * L1.normalize(), color="C1", ls="--", lw=1.0, label="(re)normalized")
ax1.set_title("A simple Line")
ax1.legend()
#plt.show()
#------------------------------------------------------------------------------

### Create a SqE model from the three lines
### Add the three weighted lines individually
icalcstrat = InelasticCalcStrategy(T=20.)
qcalcstrat = QuasielasticCalcStrategy()
# This creates the sqe model
sqe1 = SqE_kf(lines=(L1, L2, L3), lam=wavelength_from_energy(6.0), dlam=0.12, lSD=3.43, T=20)

# Sum of the Lines (l1, L2, L3) weights
sum_of_weights = sum([l.line_params["weight"] for l in (L1, L2, L3)])

f2 = plt.figure(figsize=(6.0,4.0))
ax2 = f2.add_subplot(111)
ax2.plot(e, sqe1(e), color="C4", label="S(q,E)", ls="-", lw=2.0)
ax2.plot(e, qcalcstrat.calc(L1, e) / sum_of_weights, color="C0", ls="--", label=f"{L1.name} (quasiel)")
ax2.plot(e, qcalcstrat.calc(L2, e) / sum_of_weights, color="C1", ls="--", label=f"{L2.name} (quasiel)")
# this is th inelastic line
# --> evaluation with the i(nelastic)calcstrat
ax2.plot(e, icalcstrat.calc(L3, e) / sum_of_weights, color="C2", ls="--", label=f"{L2.name} (inel)")
ax2.legend()
plt.show()

### Now we need the correction factors and the transformer!

### Also the mechanic of line.within_domain needs to be changed
### to return 0 outside of the domain and not to swallow up the points