from .DSLR import DSLR
from .ELGAN import ELGAN
from .LLFormer import LLFormer
from .RetinexFormer import RetinexFormer
from .RetinexNet import RetinexNet
from .SGM import SGM
from .ZeroDCE import ZeroDCE
from .ZeroDCEPP import ZeroDCEPP
from .RUAS import RUAS

model_registry = {
    "RetinexNet": RetinexNet,
    'DSLR': DSLR,
    'ELGAN': ELGAN,
    'ZeroDCE': ZeroDCE,
    'SGM': SGM,
    'ZeroDCEPP': ZeroDCEPP,
    'LLFormer': LLFormer,
    'RetinexFormer': RetinexFormer,
    'RUAS': RUAS
}
