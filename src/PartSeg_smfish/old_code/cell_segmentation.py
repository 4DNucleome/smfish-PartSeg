import operator
import typing
from abc import ABC
from copy import deepcopy
from typing import Callable

import numpy as np
import SimpleITK as sitk
import skimage.filters

from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, SegmentationProfile
from PartSegCore.channel_class import Channel
from PartSegCore.image_operations import gaussian
from PartSegCore.segmentation.algorithm_base import AdditionalLayerDescription, SegmentationResult
from PartSegCore.segmentation.noise_filtering import noise_filtering_dict
from PartSegCore.segmentation.segmentation_algorithm import StackAlgorithm, close_small_holes
from PartSegCore.segmentation.threshold import BaseThreshold, threshold_dict
from PartSegCore.segmentation.watershed import BaseWatershed, sprawl_dict


class SingleLayerBase(AlgorithmDescribeBase, ABC):
    @classmethod
    def calculate_layer(cls, array: np.ndarray, axis: str, parameters: dict) -> np.ndarray:
        raise NotImplementedError()


class MaxProjection(SingleLayerBase):
    @classmethod
    def calculate_layer(cls, array: np.ndarray, axis: str, parameters: dict) -> np.ndarray:
        return np.max(array, axis=axis.index("Z"))

    @classmethod
    def get_name(cls) -> str:
        return "Maximum projection"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return []


class MidArrayProjection(SingleLayerBase):
    @classmethod
    def calculate_layer(cls, array: np.ndarray, axis: str, parameters: dict) -> np.ndarray:
        slices = [slice(None) for _ in range(array.ndim)]
        z_axis = axis.index("Z")
        index = array.shape[z_axis] // 2
        slices[z_axis] = index
        return array[tuple(slices)]

    @classmethod
    def get_name(cls) -> str:
        return "Mid layer"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return []


select_layer = {
    MaxProjection.get_name(): MaxProjection,
    MidArrayProjection.get_name(): MidArrayProjection,
}


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
            AlgorithmProperty(
                "layer",
                "Layer method",
                next(iter(select_layer.keys())),
                possible_values=select_layer,
                property_type=AlgorithmDescribeBase,
            ),
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
        nucleus_projection = select_layer[self.new_parameters["layer"]["name"]].calculate_layer(
            nucleus_channel, self.image.return_order.replace("C", ""), self.new_parameters["layer"]["values"]
        )
        gauss = gaussian(nucleus_projection, self.new_parameters["nuc_gauss"])
        edges = skimage.filters.sobel(gauss)
        threshold_algorithm: BaseThreshold = threshold_dict[self.new_parameters["nuc_threshold"]["name"]]
        mask, self.nuc_thr_info = threshold_algorithm.calculate_mask(
            edges, None, self.new_parameters["nuc_threshold"]["values"], operator.ge
        )
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
                "projection nucleus": AdditionalLayerDescription(data=nucleus_projection, layer_type="image"),
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
        return (
            ["Nucleus"]
            + super().get_fields()
            + ["Cell"]
            + [
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
                ),
            ]
        )

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        if self.image is None:
            raise ValueError("image not set")
        res = super().calculation_run(report_fun)
        cell_channel = self.image.get_channel(self.new_parameters["cell"])
        cell_projection = select_layer[self.new_parameters["layer"]["name"]].calculate_layer(
            cell_channel, self.image.return_order.replace("C", ""), self.new_parameters["layer"]["values"]
        )
        gauss = gaussian(cell_projection, self.new_parameters["cell_gauss"])
        edges = skimage.filters.sobel(gauss)
        threshold_algorithm: BaseThreshold = threshold_dict[self.new_parameters["cell_threshold"]["name"]]
        mask, thr_val = threshold_algorithm.calculate_mask(
            edges, None, self.new_parameters["cell_threshold"]["values"], operator.ge
        )
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
        for i in range(1, components_num + 1):
            closed = close_small_holes(new_segment == i, self.new_parameters["close_holes_size"])
            new_segment[closed > 0] = i

        segmentation = np.concatenate([new_segment for _ in range(self.image.layers)])
        return SegmentationResult(
            segmentation=segmentation,
            parameters=self.get_segmentation_profile(),
            additional_layers={
                "projection cell": AdditionalLayerDescription(data=cell_projection, layer_type="image"),
                "edges cell": AdditionalLayerDescription(data=gauss, layer_type="image"),
                "mask": AdditionalLayerDescription(data=mask, layer_type="labels"),
                "new_segment": AdditionalLayerDescription(data=new_segment, layer_type="labels"),
                **res.additional_layers,
            },
        )


class SMAlgorithmSelect(StackAlgorithm):
    def __init__(self):
        super().__init__()
        self.nuc_thr_info = 0

    def get_info_text(self):
        return ""

    def get_segmentation_profile(self) -> SegmentationProfile:
        return SegmentationProfile("", self.get_name(), deepcopy(self.new_parameters))

    @classmethod
    def get_name(cls) -> str:
        return "layer segmentation"

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty(
                "layer",
                "Layer number",
                0,
            ),
            AlgorithmProperty("nucleus", "Nucleus channel", 0, property_type=Channel),
            AlgorithmProperty(
                "noise_filtering_nuc",
                "Filter nucleus",
                next(iter(noise_filtering_dict.keys())),
                possible_values=noise_filtering_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "nuc_threshold",
                "Nuc Threshold",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("close_holes_size", "Maximum holes size (px)", 500, (0, 10 ** 5), 10),
            AlgorithmProperty("minimum_size", "Minimum size (px)", 500, (0, 10 ** 6), 1000),
            AlgorithmProperty("cell", "Cell channel", 0, property_type=Channel),
            AlgorithmProperty(
                "noise_filtering_cell",
                "Filter cell",
                next(iter(noise_filtering_dict.keys())),
                possible_values=noise_filtering_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "cell_threshold",
                "Cell Threshold",
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
            ),
        ]

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        if self.image is None:
            raise ValueError("image not set")
        nucleus_channel = self.image.get_channel(self.new_parameters["nucleus"])
        nucleus_image = noise_filtering_dict[self.new_parameters["noise_filtering_nuc"]["name"]].noise_filter(
            nucleus_channel, self.image.spacing, self.new_parameters["noise_filtering_nuc"]["values"]
        )
        nucleus_projection = self.calculate_layer(
            nucleus_image, self.image.return_order.replace("C", ""), self.new_parameters["layer"]
        )
        threshold_algorithm: BaseThreshold = threshold_dict[self.new_parameters["nuc_threshold"]["name"]]
        mask, self.nuc_thr_info = threshold_algorithm.calculate_mask(
            nucleus_projection, None, self.new_parameters["nuc_threshold"]["values"], operator.ge
        )
        print("Threshold nucleus: ", self.nuc_thr_info)
        mask = close_small_holes(mask, self.new_parameters["close_holes_size"])
        core_objects = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(sitk.GetImageFromArray(mask), True),
                self.new_parameters["minimum_size"],
            )
        )
        cell_channel = self.image.get_channel(self.new_parameters["cell"])
        cell_image = noise_filtering_dict[self.new_parameters["noise_filtering_cell"]["name"]].noise_filter(
            cell_channel, self.image.spacing, self.new_parameters["noise_filtering_cell"]["values"]
        )
        nucleus_projection = self.calculate_layer(
            cell_image, self.image.return_order.replace("C", ""), self.new_parameters["layer"]
        )
        threshold_algorithm: BaseThreshold = threshold_dict[self.new_parameters["cell_threshold"]["name"]]
        mask_cell, thr_val = threshold_algorithm.calculate_mask(
            nucleus_projection, None, self.new_parameters["cell_threshold"]["values"], operator.ge
        )
        path_sprawl: BaseWatershed = sprawl_dict[self.new_parameters["sprawl_type"]["name"]]
        components_num = np.max(core_objects)
        print("cc", components_num)
        max_val = np.max(nucleus_projection[(mask > 0) * (core_objects == 0)])
        new_segment = path_sprawl.sprawl(
            mask_cell,
            core_objects,
            nucleus_projection,
            components_num,
            self.image.spacing[1:],
            True,
            operator.ge,
            self.new_parameters["sprawl_type"]["values"],
            thr_val,
            max_val,
        )
        for i in range(1, components_num + 1):
            closed = close_small_holes(new_segment == i, self.new_parameters["close_holes_size"])
            new_segment[closed > 0] = i

        segmentation = np.concatenate([new_segment for _ in range(self.image.layers)])
        return SegmentationResult(
            segmentation=segmentation,
            parameters=self.get_segmentation_profile(),
            additional_layers={
                "segment": AdditionalLayerDescription(data=new_segment, layer_type="image"),
                "nucleus_projection": AdditionalLayerDescription(data=nucleus_projection, layer_type="image"),
                "mask": AdditionalLayerDescription(data=mask, layer_type="labels"),
                "new_segment": AdditionalLayerDescription(data=new_segment, layer_type="labels"),
            },
        )

    def calculate_layer(cls, array: np.ndarray, axis: str, layer_num: int) -> np.ndarray:
        slices: typing.List[typing.Union[int, slice]] = [slice(None) for _ in range(array.ndim)]
        z_axis = axis.index("Z")
        slices[z_axis] = layer_num
        return array[tuple(slices)]
