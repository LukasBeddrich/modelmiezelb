from modelmiezelb.lineshape import LorentzianLine
from modelmiezelb.sqe_model import SqE, SqE_from_arg
from modelmiezelb.transformer import SqtTransformer
from modelmiezelb.fit import FitModelCreator
from modelmiezelb.utils.helpers import format_param_dict_for_logger, format_sqt_lines_for_logger
###############################################################################

### Creating a SqE model for transformation
# We need some lines
L1 = LorentzianLine("Lorentzian1", (-5.0, 5.0), x0=0.0, width=0.4, c=0.0, weight=2)
L2 = LorentzianLine(name="Lorentzian2", domain=(-5.0, 5.0), x0=-1.0, width=0.4, c=0.0, weight=1)
# Contruct a SqE model
sqe = SqE(lines=(L1, L2), lam=6.0, dlam=0.12, lSD=3.43, T=20)
### Instantiate a transformer
sqt = SqtTransformer(sqe, nlam=20, ne=10000)

def test_format_param_dict_for_logger():
    names = FitModelCreator.get_param_names(sqe)
    params = [0.020, 0.0, 12, 0.15, 0.020, 0.0, 1]
    print(format_param_dict_for_logger(dict(zip(names, params))))

#------------------------------------------------------------------------------

def test_format_sqt_lines_for_logger():
    print(format_sqt_lines_for_logger(sqt))

if __name__ == "__main__":
    test_format_param_dict_for_logger()
    test_format_sqt_lines_for_logger()