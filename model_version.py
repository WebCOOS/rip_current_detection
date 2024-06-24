from enum import Enum


class ModelFramework(str, Enum):
    TORCHVISION = "TORCHVISION"
    ULTRALYTICS = "ULTRALYTICS"


class TorchvisionModelName(str, Enum):
    rip_current_detector = "rip_current_detector"


class TorchvisionModelVersion(str, Enum):
    one = "1"


class YOLOModelName(str, Enum):
    ripdetect_walton = "ripdetect_walton"


class YOLOModelVersion(str, Enum):
    yolov8x_11 = "yolov8x_1.1"
