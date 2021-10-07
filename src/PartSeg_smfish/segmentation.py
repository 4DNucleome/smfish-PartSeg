import operator
import typing
from copy import deepcopy
from typing import Callable, List, Tuple, Union

import numpy as np
import SimpleITK
from napari.layers import Image, Labels
from napari.types import LayerDataTuple

from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, ROIExtractionProfile
from PartSegCore.channel_class import Channel
from PartSegCore.convex_fill import convex_fill
from PartSegCore.image_operations import gaussian
from PartSegCore.segmentation import ROIExtractionAlgorithm
from PartSegCore.segmentation.algorithm_base import AdditionalLayerDescription, ROIExtractionResult, SegmentationResult
from PartSegCore.segmentation.noise_filtering import DimensionType, GaussNoiseFiltering, noise_filtering_dict
from PartSegCore.segmentation.segmentation_algorithm import CellFromNucleusFlow, StackAlgorithm
from PartSegCore.segmentation.threshold import BaseThreshold, threshold_dict
from PartSegImage import Image as PSImage


class SMSegmentationBase(ROIExtractionAlgorithm):
    @classmethod
    def support_time(cls):
        return False

    @classmethod
    def support_z(cls):
        return True

    def background_estimate(self, channel: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        channel_nuc = self.get_channel(self.new_parameters["channel_nuc"])
        noise_filtering_parameters = self.new_parameters["noise_filtering_nucleus"]
        cleaned_image = noise_filtering_dict[noise_filtering_parameters["name"]].noise_filter(
            channel_nuc, self.image.spacing, noise_filtering_parameters["values"]
        )
        thr: BaseThreshold = threshold_dict[self.new_parameters["nucleus_threshold"]["name"]]
        nucleus_mask, nucleus_thr_val = thr.calculate_mask(
            cleaned_image, self.mask, self.new_parameters["nucleus_threshold"]["values"], operator.ge
        )
        nucleus_connect = SimpleITK.ConnectedComponent(SimpleITK.GetImageFromArray(nucleus_mask), True)
        nucleus_segmentation = SimpleITK.GetArrayFromImage(
            SimpleITK.RelabelComponent(nucleus_connect, self.new_parameters["minimum_nucleus_size"])
        )
        nucleus_segmentation = convex_fill(nucleus_segmentation)

        channel_molecule = self.get_channel(self.new_parameters["channel_molecule"])
        estimated = self.background_estimate(channel_molecule)
        if self.mask is not None:
            estimated[self.mask == 0] = 0
        thr: BaseThreshold = threshold_dict[self.new_parameters["molecule_threshold"]["name"]]
        molecule_mask, molecule_thr_val = thr.calculate_mask(
            estimated, self.mask, self.new_parameters["molecule_threshold"]["values"], operator.ge
        )
        nucleus_connect = SimpleITK.ConnectedComponent(SimpleITK.GetImageFromArray(molecule_mask), True)

        molecule_segmentation = SimpleITK.GetArrayFromImage(
            SimpleITK.RelabelComponent(nucleus_connect, self.new_parameters["minimum_molecule_size"])
        )

        sizes = np.bincount(molecule_segmentation.flat)
        elements = np.unique(molecule_segmentation[molecule_segmentation > 0])

        cellular_components = set(np.unique(molecule_segmentation[nucleus_segmentation == 0]))
        if 0 in cellular_components:
            cellular_components.remove(0)
        nucleus_components = set(np.unique(molecule_segmentation[nucleus_segmentation > 0]))
        if 0 in nucleus_components:
            nucleus_components.remove(0)
        mixed_components = cellular_components & nucleus_components
        cellular_components = cellular_components - mixed_components
        nucleus_components = nucleus_components - mixed_components
        label_types = {}
        label_types.update({i: "Nucleus" for i in nucleus_components})
        label_types.update({i: "Cytoplasm" for i in cellular_components})
        label_types.update({i: "Mixed" for i in mixed_components})

        annotation = {el: {"voxels": sizes[el], "type": label_types[el], "number": el} for el in elements}
        position_masking = np.zeros(
            (np.max(elements) if elements.size > 0 else 0) + 1, dtype=molecule_segmentation.dtype
        )
        for el in cellular_components:
            position_masking[el] = 1
        for el in mixed_components:
            position_masking[el] = 2
        for el in nucleus_components:
            position_masking[el] = 3
        position_array = position_masking[molecule_segmentation]

        return SegmentationResult(
            roi=molecule_segmentation,
            parameters=self.get_segmentation_profile(),
            additional_layers={
                "nucleus segmentation": AdditionalLayerDescription(data=nucleus_segmentation, layer_type="labels"),
                "roi segmentation": AdditionalLayerDescription(data=molecule_segmentation, layer_type="labels"),
                "estimated signal": AdditionalLayerDescription(data=estimated, layer_type="image"),
                "channel molecule": AdditionalLayerDescription(data=channel_molecule, layer_type="image"),
                "position": AdditionalLayerDescription(data=position_array, layer_type="labels"),
            },
            roi_annotation=annotation,
            alternative_representation={"position": position_array},
        )

    def get_info_text(self):
        return ""

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        return ROIExtractionProfile("", self.get_name(), deepcopy(self.new_parameters))

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return [
            AlgorithmProperty("channel_nuc", "Nucleus Channel", 0, value_type=Channel),
            AlgorithmProperty(
                "noise_filtering_nucleus",
                "Filter nucleus",
                next(iter(noise_filtering_dict.keys())),
                possible_values=noise_filtering_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "nucleus_threshold",
                "Nucleus Threshold",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("minimum_nucleus_size", "Minimum nucleus size (px)", 500, (0, 10 ** 6), 1000),
            AlgorithmProperty("channel_molecule", "Channel molecule", 1, value_type=Channel),
            AlgorithmProperty(
                "molecule_threshold",
                "Molecule Threshold",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("minimum_molecule_size", "Minimum molecule size (px)", 5, (0, 10 ** 6), 1000),
            AlgorithmProperty("normalize", "Normalize", False),
        ]


class SMSegmentation(SMSegmentationBase):
    @classmethod
    def get_name(cls) -> str:
        return "sm-fish segmentation"

    def background_estimate(self, channel: np.ndarray) -> np.ndarray:
        mask = self.mask if self.mask is not None else channel > 0
        gaussian = _gauss_background_estimate(
            channel,
            mask,
            self.image.spacing,
            self.new_parameters["background_estimate_radius"],
            self.new_parameters["foreground_estimate_radius"],
        )
        if self.new_parameters["normalize"]:
            shift = np.mean(gaussian)
            std = np.std(gaussian)
            return (gaussian - shift) / (std)
        return gaussian

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        res = super().get_fields()
        res.insert(
            -3,
            AlgorithmProperty(
                "background_estimate_radius",
                "Background estimate radius",
                5.0,
                (0, 20),
            ),
        )
        res.insert(
            -3,
            AlgorithmProperty(
                "foreground_estimate_radius",
                "Foreground estimate radius",
                2.5,
                (0, 20),
            ),
        )
        return res


class SMLaplacianSegmentation(SMSegmentationBase):
    @classmethod
    def get_name(cls) -> str:
        return "sm-fish laplacian segmentation"

    def background_estimate(self, channel: np.ndarray) -> np.ndarray:
        mask = self.mask if self.mask is not None else channel > 0
        laplacian = _laplacian_estimate(channel, mask, self.new_parameters["laplacian_radius"])
        if self.new_parameters["normalize"]:
            shift = np.mean(laplacian)
            std = np.std(laplacian)
            return (laplacian - shift) / (std)
        return laplacian

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        res = super().get_fields()
        res.insert(
            -3,
            AlgorithmProperty(
                "laplacian_radius",
                "Laplacian radius",
                1.3,
                (0, 20),
            ),
        )
        return res


def gauss_background_estimate(
    image: Image,
    mask: Labels,
    background_gauss_radius: float = 5,
    foreground_gauss_radius: float = 2.5,
    clip_bellow_0: bool = True,
) -> LayerDataTuple:
    # process the image
    mask = mask.data if mask is not None else np.ones(image.data.shape, dtype=np.uint8)
    resp = _gauss_background_estimate(
        image.data[0], mask[0], image.scale[1:], background_gauss_radius, foreground_gauss_radius
    )
    if clip_bellow_0:
        resp[resp < 0] = 0
    resp = resp.reshape((1,) + resp.shape)
    # return it + some layer properties
    return LayerDataTuple((resp, {"colormap": "gray", "scale": image.scale, "name": "Signal estimate"}))


def _gauss_background_estimate(
    channel: np.ndarray,
    mask: np.ndarray,
    scale: Union[List[float], Tuple[Union[float, int]]],
    background_gauss_radius: float,
    foreground_gauss_radius: float,
) -> np.ndarray:
    data = channel.astype(np.float64)
    mean_background = np.mean(data[mask > 0])
    data[mask == 0] = mean_background
    data = gaussian(data, 15, False)
    data[mask > 0] = channel[mask > 0]
    background_estimate = GaussNoiseFiltering.noise_filter(
        data, scale, {"dimension_type": DimensionType.Layer, "radius": background_gauss_radius}
    )
    foreground_estimate = GaussNoiseFiltering.noise_filter(
        data, scale, {"dimension_type": DimensionType.Layer, "radius": foreground_gauss_radius}
    )
    resp = foreground_estimate - background_estimate
    resp[mask == 0] = 0
    return resp


def laplacian_estimate(image: Image, mask: Labels, radius=1.30, clip_bellow_0=True) -> LayerDataTuple:
    mask = mask.data if mask is not None else np.ones(image.data.shape, dtype=np.uint8)
    res = _laplacian_estimate(image.data[0], mask[0], radius=radius)
    if clip_bellow_0:
        res[res < 0] = 0
    res = res.reshape(image.data.shape)
    return LayerDataTuple((res, {"colormap": "magma", "scale": image.scale, "name": "Laplacian estimate"}))


def _laplacian_estimate(channel: np.ndarray, mask: np.ndarray, radius=1.30) -> np.ndarray:
    data = channel.astype(np.float64)
    mean_background = np.mean(data[mask > 0])
    data[mask == 0] = mean_background
    data = gaussian(data, 15, False)
    data[mask > 0] = channel[mask > 0]
    return -SimpleITK.GetArrayFromImage(SimpleITK.LaplacianRecursiveGaussian(SimpleITK.GetImageFromArray(data), radius))


def laplacian_check(image: Image, mask: Labels, radius=1.0, threshold=10.0, min_size=50) -> LayerDataTuple:
    data = image.data[0]
    laplaced = -SimpleITK.GetArrayFromImage(
        SimpleITK.LaplacianRecursiveGaussian(SimpleITK.GetImageFromArray(data), radius)
    )

    labeling = SimpleITK.GetArrayFromImage(
        SimpleITK.RelabelComponent(
            SimpleITK.ConnectedComponent(
                SimpleITK.BinaryThreshold(SimpleITK.GetImageFromArray(laplaced), threshold, float(laplaced.max())),
                SimpleITK.GetImageFromArray(mask.data[0]),
            ),
            min_size,
        )
    )
    labeling = labeling.reshape((1,) + data.shape)
    return LayerDataTuple((labeling, {"scale": image.scale, "name": "Signal estimate"}))


class LayerRangeThresholdFlow(StackAlgorithm):
    def get_info_text(self):
        return ""

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        return ROIExtractionProfile("", self.get_name(), deepcopy(self.new_parameters))

    @classmethod
    def get_name(cls) -> str:
        return "Maximum projection Threshold Flow"

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty("lower_layer", "Lower Layer", 0, options_range=(-1, 1000)),
            AlgorithmProperty("upper_layer", "Upper Layer", -1, options_range=(-1, 1000)),
        ] + CellFromNucleusFlow.get_fields()

    @staticmethod
    def get_steps_num():
        return CellFromNucleusFlow.get_steps_num() + 2

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> ROIExtractionResult:
        count = [0]

        def report_fun_wrap(text, num):
            report_fun(text, num + 1)
            count[0] = num + 1

        report_fun("Maximum projection", 0)
        lower_layer = self.new_parameters["lower_layer"]
        upper_layer = self.new_parameters["upper_layer"]
        if upper_layer < 0:
            upper_layer = self.image.shape[self.image.stack_pos] - upper_layer
        upper_layer += 1
        slice_arr = [slice(None) for _ in self.image.axis_order]
        slice_arr[self.image.stack_pos] = slice(lower_layer, upper_layer)
        slice_arr.pop(self.image.channel_pos)
        image: PSImage = self.image.substitute(mask=self.mask).cut_image(slice_arr, frame=0)

        new_data = np.max(image.get_data(), axis=image.stack_pos)
        mask = np.min(image.mask, axis=image.get_array_axis_positions()["Z"]) if image.mask is not None else None
        image = PSImage(new_data, image.spacing[1:], mask=mask, axes_order=image.axis_order.replace("Z", ""))

        segment_method = CellFromNucleusFlow()
        segment_method.set_image(image)
        segment_method.set_mask(image.mask)
        parameters = dict(self.new_parameters)
        del parameters["lower_layer"]
        del parameters["upper_layer"]
        segment_method.set_parameters(**parameters)
        partial_res = segment_method.calculation_run(report_fun_wrap)

        report_fun("Copy layers", count[0] + 1)
        res_roi = np.zeros(self.image.get_channel(0).shape, dtype=partial_res.roi.dtype)
        base_index = (slice(None),) * (self.image.stack_pos)
        for i in range(lower_layer, upper_layer):
            res_roi[base_index + (i,)] = partial_res.roi

        report_fun("Prepare result", count[0] + 2)
        additional_layer = {
            "maximum_projection": AdditionalLayerDescription(new_data, "image", "maximum projection"),
            **partial_res.additional_layers,
        }

        return ROIExtractionResult(
            roi=res_roi, parameters=self.get_segmentation_profile(), additional_layers=additional_layer
        )


def maximum_projection(
    image: Image, lower_layer: int = 0, upper_layer: int = 1, axis_num: int = PSImage.get_array_axis_positions()["Z"]
) -> LayerDataTuple:
    data = image.data
    slice_arr = [slice(None) for _ in data.shape]
    if upper_layer < 0:
        upper_layer = data.shape[axis_num] - upper_layer
    upper_layer += 1
    slice_arr[axis_num] = slice(lower_layer, upper_layer)
    res = np.max(data[tuple(slice_arr)], axis=axis_num)
    shape = list(data.shape)
    shape[axis_num] = 1
    return LayerDataTuple(
        (res.reshape(shape), {"colormap": "magma", "scale": image.scale, "name": "Maximum projection"})
    )
