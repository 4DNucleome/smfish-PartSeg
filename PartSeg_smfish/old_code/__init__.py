from PartSegCore.register import RegisterEnum
from PartSegCore.register import register as _register

from .cell_segmentation import SMAlgorithmCell, SMAlgorithmNuc, SMAlgorithmSelect
from .peak_segment import PeakSegment
from .threshold import ScaledOtsu


def register():
    _register(SMAlgorithmCell, RegisterEnum.mask_algorithm)
    _register(SMAlgorithmNuc, RegisterEnum.mask_algorithm)
    _register(SMAlgorithmSelect, RegisterEnum.mask_algorithm)
    _register(ScaledOtsu, RegisterEnum.threshold)
    _register(PeakSegment, RegisterEnum.analysis_algorithm)
