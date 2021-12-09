import sys

from napari_plugin_engine import napari_hook_implementation

from . import measurement, segmentation
from .copy_labels import CopyLabelWidget
from .segmentation import gauss_background_estimate, laplacian_check, laplacian_estimate, maximum_projection
from .verify_points import find_single_points, verify_segmentation

if "reload" in globals():
    import importlib

    importlib.reload(segmentation)

reload = False

_hiddentimports = ["sklearn.neighbors._partition_nodes"]


def register():
    from PartSegCore.register import RegisterEnum
    from PartSegCore.register import register as register_fun

    register_fun(segmentation.SMSegmentationBase, RegisterEnum.roi_analysis_segmentation_algorithm)
    register_fun(segmentation.LayerRangeThresholdFlow, RegisterEnum.roi_mask_segmentation_algorithm)
    register_fun(measurement.ComponentType, RegisterEnum.analysis_measurement)

    if getattr(sys, "frozen", False):
        import napari

        napari.plugins.plugin_manager.register(sys.modules[__name__])


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return CopyLabelWidget


@napari_hook_implementation
def napari_experimental_provide_function():
    return gauss_background_estimate  # , {"area": "bottom"}


@napari_hook_implementation(specname="napari_experimental_provide_function")
def napari_experimental_provide_function2():
    return laplacian_check


@napari_hook_implementation(specname="napari_experimental_provide_function")
def napari_experimental_provide_function3():
    return laplacian_estimate


@napari_hook_implementation(specname="napari_experimental_provide_function")
def napari_experimental_provide_function4():
    return maximum_projection


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def napari_experimental_provide_dock_widget2():
    return verify_segmentation, {"name": "Verify Segmentation"}


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def napari_experimental_provide_dock_widget3():
    return find_single_points, {"name": "Single points"}


try:
    from .big_fish import spot_detect_big_fish

    @napari_hook_implementation(specname="napari_experimental_provide_function")
    def napari_experimental_provide_function5():
        return spot_detect_big_fish


except ImportError:
    pass
