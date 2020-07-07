import operator
from copy import deepcopy
from typing import Callable

import numpy as np
import skimage.filters
import SimpleITK as sitk

from PartSegCore.algorithm_describe_base import AlgorithmProperty, AlgorithmDescribeBase, SegmentationProfile
from PartSegCore.channel_class import Channel
from PartSegCore.image_operations import gaussian
from PartSegCore.segmentation.algorithm_base import SegmentationResult, AdditionalLayerDescription
from PartSegCore.segmentation.segmentation_algorithm import StackAlgorithm, close_small_holes
from PartSegCore.segmentation.threshold import threshold_dict, BaseThreshold
from PartSegCore.segmentation.watershed import sprawl_dict, BaseWatershed


class SMAlgorithmNuc(StackAlgorithm):
    def __init__(self):
        super().__init__()
        self.nuc_thr_info = 0

    def get_info_text(self):
        return ""

    def get_segmentation_profile(self) -> SegmentationProfile:
        return SegmentationProfile("", self.get_name(), deepcopy(self.new_parameters))

    @classmethod
    def get_name(cls) -> str:
        return "Projection segmentation"

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty("nucleus", "Nucleus channel", 0, property_type=Channel),
            AlgorithmProperty("nuc_gauss", "Filtering radius", 3, [0, 20]),
            AlgorithmProperty(
                "nuc_threshold",
                "Threshold",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("close_holes_size", "Maximum holes size (px)", 500, (0, 10 ** 5), 10),
            AlgorithmProperty("minimum_size", "Minimum size (px)", 500, (0, 10 ** 6), 1000),

        ]

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        if self.image is None:
            raise ValueError("image not set")
        nucleus_channel = self.image.get_channel(self.new_parameters["nucleus"])
        max_nucleus_projection = np.max(nucleus_channel, axis=self.image.stack_pos)
        gauss = gaussian(max_nucleus_projection, self.new_parameters["nuc_gauss"])
        edges = skimage.filters.sobel(gauss)
        threshold_algorithm: BaseThreshold = threshold_dict[self.new_parameters["nuc_threshold"]["name"]]
        mask, self.nuc_thr_info = threshold_algorithm.calculate_mask(edges, None, self.new_parameters["nuc_threshold"]["values"], operator.ge)
        print("Threshold nucleus: ", self.nuc_thr_info)
        mask = close_small_holes(mask, self.new_parameters["close_holes_size"])
        core_objects = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(sitk.GetImageFromArray(mask), True),
                self.new_parameters["minimum_size"],
            )
        )
        segmentation = np.concatenate([core_objects for _ in range(self.image.layers)])
        return SegmentationResult(
            segmentation=segmentation,
            parameters=self.get_segmentation_profile(),
            additional_layers={
                "projection nucleus": AdditionalLayerDescription(data=max_nucleus_projection, layer_type="image"),
                "edges nucleus": AdditionalLayerDescription(data=gauss, layer_type="image"),
                "core_objects": AdditionalLayerDescription(data=core_objects, layer_type="labels"),
            },
        )

class SMAlgorithmCell(SMAlgorithmNuc):
    def __init__(self):
        super().__init__()
        self.cell_thr_info = 0

    @classmethod
    def get_name(cls) -> str:
        return "Projection cell segmentation"

    @classmethod
    def get_fields(cls):
        return ["Nucleus"] + super().get_fields() + ["Cell"] + \
        [
            AlgorithmProperty("cell", "Cell channel", 0, property_type=Channel),
            AlgorithmProperty("cell_gauss", "Filtering radius", 5, [0, 20]),
            AlgorithmProperty(
                "cell_threshold",
                "Threshold",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "sprawl_type",
                "Flow type",
                next(iter(sprawl_dict.keys())),
                possible_values=sprawl_dict,
                property_type=AlgorithmDescribeBase,
            )
        ]

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        if self.image is None:
            raise ValueError("image not set")
        res = super().calculation_run(report_fun)
        cell_channel = self.image.get_channel(self.new_parameters["cell"])
        max_cell_projection = np.max(cell_channel, axis=self.image.stack_pos)
        gauss = gaussian(max_cell_projection, self.new_parameters["cell_gauss"])
        edges = skimage.filters.sobel(gauss)
        threshold_algorithm: BaseThreshold = threshold_dict[self.new_parameters["cell_threshold"]["name"]]
        mask, thr_val = threshold_algorithm.calculate_mask(edges, None, self.new_parameters["cell_threshold"]["values"], operator.ge)
        print("Threshold cell: ", thr_val)
        path_sprawl: BaseWatershed = sprawl_dict[self.new_parameters["sprawl_type"]["name"]]
        core_objects = res.additional_layers["core_objects"].data
        components_num = np.max(core_objects)
        print("cc", components_num)
        max_val = np.max(gauss[(mask > 0) * (core_objects == 0)])
        new_segment = path_sprawl.sprawl(
            mask,
            core_objects,
            gauss,
            components_num,
            self.image.spacing[1:],
            True,
            operator.ge,
            self.new_parameters["sprawl_type"]["values"],
            thr_val,
            max_val,
        )

        segmentation = np.concatenate([new_segment for _ in range(self.image.layers)])
        return SegmentationResult(
            segmentation=segmentation,
            parameters=self.get_segmentation_profile(),
            additional_layers={
                "projection cell": AdditionalLayerDescription(data=max_cell_projection, layer_type="image"),
                "edges cell": AdditionalLayerDescription(data=gauss, layer_type="image"),
                "mask": AdditionalLayerDescription(data=mask, layer_type="labels"),
                "new_segment": AdditionalLayerDescription(data=new_segment, layer_type="labels"),
                **res.additional_layers
            },
        )

