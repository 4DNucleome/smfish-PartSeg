import typing
from typing import Callable

from bigfish import detection
from napari.layers import Image
from napari.types import LayerDataTuple

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


def spot_detect(image: Image, psf_z: float = 400, psf_yx: float = 100) -> LayerDataTuple:
    spots, threshold = detection.detect_spots(
        image.data.squeeze(),
        return_threshold=True,
        voxel_size_z=image.scale[1],
        voxel_size_yx=image.scale[2],
        psf_z=psf_z,
        psf_yx=psf_yx,
    )
    print(threshold)
    return LayerDataTuple((spots, {"name": "Spots", "scale": image.scale[1:]}, "points"))
