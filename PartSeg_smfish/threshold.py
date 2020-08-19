import typing

import numpy as np
import skimage.filters


from PartSegCore.algorithm_describe_base import AlgorithmProperty
from PartSegCore.segmentation.threshold import BaseThreshold


class ScaledOtsu(BaseThreshold):
    @classmethod
    def calculate_mask(
        cls,
        data: np.ndarray,
        mask: typing.Optional[np.ndarray],
        arguments: dict,
        operator: typing.Callable[[object, object], bool],
    ):
        if arguments["masked"] and mask is not None:
            otsu_value = skimage.filters.threshold_otsu(data[mask > 0])
        else:
            otsu_value = skimage.filters.threshold_otsu(data)
        otsu_value *= arguments["scale"]
        res = operator(data, otsu_value).astype(np.uint8)
        if arguments["masked"] and mask is not None:
            res[mask == 0] = 0
        return res, otsu_value

    @classmethod
    def get_name(cls) -> str:
        return "Scaled Otsu"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return [
            AlgorithmProperty("scale", "Scale ratio", 1.0, [0, 5], single_steep=0.1),
            AlgorithmProperty("masked", "Apply mask", True),
        ]
