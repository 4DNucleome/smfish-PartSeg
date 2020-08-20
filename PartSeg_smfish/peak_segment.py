import dataclasses
import operator
import typing
from copy import deepcopy
from typing import Callable

import numpy as np
import SimpleITK

from PartSegCore.algorithm_describe_base import AlgorithmProperty, AlgorithmDescribeBase
from PartSegCore.channel_class import Channel
from PartSegCore.image_operations import gaussian
from PartSegCore.segmentation import RestartableAlgorithm
from PartSegCore.segmentation.algorithm_base import SegmentationResult, AdditionalLayerDescription
from PartSegCore.segmentation.threshold import threshold_dict, BaseThreshold
from PartSegCore.utils import bisect


class PeakSegment(RestartableAlgorithm):
    def __init__(self):
        super().__init__()
        self.background_removed = None
        self.signal_filtered = None
        self.threshold_image = None
        self.threshold_info = 0
        self.segmentation = None
        self._sizes_array = []
        self.indexes = (0, 0)

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        restarted = False
        if self.channel is None or self.parameters["channel"] != self.new_parameters["channel"]:
            self.parameters["channel"] = self.new_parameters["channel"]
            self.channel = self.get_channel(self.new_parameters["channel"])
            restarted = True
        if restarted or self.parameters["background_gauss"] != self.new_parameters["background_gauss"]:
            self.parameters["background_gauss"] = self.new_parameters["background_gauss"]
            restarted = True
            if self.mask is not None:
                data = np.copy(self.channel)
                background_val = np.mean(self.channel[self.mask > 0])
                data[self.mask == 0] = background_val
            else:
                data = self.channel
            self.background_removed = self.channel.astype(np.float64) - gaussian(
                data, self.new_parameters["background_gauss"]
            )
        if restarted or self.parameters["signal_gauss"] != self.new_parameters["signal_gauss"]:
            self.parameters["signal_gauss"] = self.new_parameters["signal_gauss"]
            restarted = True
            self.signal_filtered = gaussian(self.background_removed, self.new_parameters["signal_gauss"])
        if restarted or self.parameters["threshold"] != self.new_parameters["threshold"]:
            print("tt")
            restarted = True
            self.parameters["threshold"] = deepcopy(self.new_parameters["threshold"])
            thr: BaseThreshold = threshold_dict[self.new_parameters["threshold"]["name"]]
            self.threshold_image, self.threshold_info = thr.calculate_mask(
                self.signal_filtered, self.mask, self.new_parameters["threshold"]["values"], operator=operator.ge
            )
            if self.threshold_image.max() == 0:
                res = self.prepare_result(self.threshold_image.astype(np.uint8))
                info_text = (
                    "Something wrong with chosen threshold. Please check it. "
                    "May be to low or to high. The channel bright range is "
                    f"{self.signal_filtered.min()}-{self.signal_filtered.max()} "
                    f"and chosen threshold is {self.threshold_info}"
                )
                return dataclasses.replace(res, info_text=info_text)
        if restarted:
            connect = SimpleITK.ConnectedComponent(SimpleITK.GetImageFromArray(self.threshold_image))
            self.segmentation = SimpleITK.GetArrayFromImage(SimpleITK.RelabelComponent(connect))
            self._sizes_array = np.bincount(self.segmentation.flat)
            restarted = True
        if restarted or self.new_parameters["minimum_size"] != self.parameters["minimum_size"] or \
                self.new_parameters["maximum_size"] != self.parameters["maximum_size"]:
            self.parameters["minimum_size"] = self.new_parameters["minimum_size"]
            self.parameters["maximum_size"] = self.new_parameters["maximum_size"]
            minimum_size = self.new_parameters["minimum_size"]
            ind = bisect(self._sizes_array[1:], minimum_size, lambda x, y: x > y)
            ind2 = bisect(self._sizes_array[1:], self.new_parameters["maximum_size"], lambda x, y: x > y)
            print("aaa", ind, ind2)
            finally_segment = np.copy(self.segmentation)
            finally_segment[finally_segment > ind] = 0
            finally_segment[finally_segment <= ind2] = 0
            finally_segment[finally_segment > 0] - (ind2-1)
            self.components_num = ind - ind2
            self.indexes = (ind2, ind)
            if ind == 0:
                info_text = (
                    "Please check the minimum size parameter. " f"The biggest element has size {self._sizes_array[1]}"
                )
            else:
                info_text = ""
            res = self.prepare_result(finally_segment)
            return dataclasses.replace(res, info_text=info_text)

    def prepare_result(self, segmentation: np.ndarray) -> SegmentationResult:
        """
        Collect data for result.

        :param segmentation: array with segmentation
        :return: algorithm result description
        """
        return SegmentationResult(
            segmentation=segmentation,
            parameters=self.get_segmentation_profile(),
            additional_layers=self.get_additional_layers(),
        )

    def get_additional_layers(
        self, full_segmentation: typing.Optional[np.ndarray] = None
    ) -> typing.Dict[str, AdditionalLayerDescription]:
        """
        Create dict with standard additional layers.

        :param full_segmentation: no size filtering if not `self.segmentation`

        :return:
        """
        if full_segmentation is None:
            full_segmentation = self.segmentation
        return {
            "background removed": AdditionalLayerDescription(data=self.background_removed, layer_type="image"),
            "signal filtered": AdditionalLayerDescription(data=self.signal_filtered, layer_type="image"),
            "no size filtering": AdditionalLayerDescription(data=full_segmentation, layer_type="labels"),
        }

    def get_info_text(self):
        return f"Threshold: {self.threshold_info}\nSizes: " + ", ".join(
            map(str, self._sizes_array[self.indexes[0]+1: self.indexes[1]+1])
        )

    @classmethod
    def get_name(cls) -> str:
        return "smFISH"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return [
            AlgorithmProperty("channel", "Channel", 0, property_type=Channel),
            AlgorithmProperty("background_gauss", "Background filter radius", 5.0, options_range=(0.1, 20)),
            AlgorithmProperty("signal_gauss", "Signal filter radius", 1.0, options_range=(0.1, 20)),
            AlgorithmProperty(
                "threshold",
                "Threshold",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("minimum_size", "Minimum size (px)", 5, (0, 10 ** 6), 5),
            AlgorithmProperty("maximum_size", "Maximum size (px)", 1000, (0, 10 ** 6), 5),
        ]
