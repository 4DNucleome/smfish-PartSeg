from PartSegCore.register import register as _register
from PartSegCore.register import RegisterEnum

from .cell_segmentation import SMAlgorithmNuc, SMAlgorithmCell, SMAlgorithmSelect
from .threshold import ScaledOtsu
from .peak_segment import PeakSegment


def register():
    _register(SMAlgorithmCell, RegisterEnum.mask_algorithm)
    _register(SMAlgorithmNuc, RegisterEnum.mask_algorithm)
    _register(SMAlgorithmSelect, RegisterEnum.mask_algorithm)
    _register(ScaledOtsu, RegisterEnum.threshold)
    _register(PeakSegment, RegisterEnum.analysis_algorithm)
