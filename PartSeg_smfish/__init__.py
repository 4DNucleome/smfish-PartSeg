from PartSegCore.register import register as _register
from PartSegCore.register import RegisterEnum

from .cell_segmentation import SMAlgorithmNuc, SMAlgorithmCell
from .threshold import ScaledOtsu


def register():
    _register(SMAlgorithmCell, RegisterEnum.mask_algorithm)
    _register(SMAlgorithmNuc, RegisterEnum.mask_algorithm)
    _register(ScaledOtsu, RegisterEnum.threshold)
