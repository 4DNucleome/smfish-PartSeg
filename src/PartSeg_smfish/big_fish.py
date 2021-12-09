import typing
from typing import Callable

from bigfish import detection, stack
from napari.layers import Image
from napari.types import LayerDataTuple
from napari.utils import progress

from PartSegCore.algorithm_describe_base import AlgorithmProperty, ROIExtractionProfile
from PartSegCore.segmentation import ROIExtractionAlgorithm, ROIExtractionResult


class BigFishSpotDetector(ROIExtractionAlgorithm):
    @classmethod
    def support_time(cls):
        return False

    @classmethod
    def support_z(cls):
        return True

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> ROIExtractionResult:
        pass

    def get_info_text(self):
        return ""

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        pass

    @classmethod
    def get_name(cls) -> str:
        return "BigFish spot detector"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return [
            AlgorithmProperty("psf_xy", "PSF xy", default_value=1),
            AlgorithmProperty("psf_z", "PSF z", default_value=1),
        ]


def _spot_detect_big_fish(image: Image, psf_z: float = 400, psf_yx: float = 100) -> LayerDataTuple:
    sigma = stack.get_sigma(image.scale[1], image.scale[2], psf_z, psf_yx)
    yield
    image_filtered = stack.log_filter(image.data.squeeze(), sigma)
    yield
    mask_local_max = detection.local_maximum_detection(image_filtered, sigma)
    yield
    threshold = detection.automated_threshold_setting(image_filtered, mask_local_max)
    yield
    spots, _ = detection.spots_thresholding(image_filtered, mask_local_max, threshold)
    yield LayerDataTuple((spots, {"name": "Spots", "scale": image.scale[1:]}, "points"))


def spot_detect_big_fish(image: Image, psf_z: float = 400, psf_yx: float = 308) -> LayerDataTuple:
    for el in progress(_spot_detect_big_fish(image, psf_z, psf_yx)):
        print(el)
    return el
