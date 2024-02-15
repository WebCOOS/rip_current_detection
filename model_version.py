from enum import Enum


class ModelFramework(str, Enum):
    TORCHVISION = "TORCHVISION"


class TorchvisionModelName(str, Enum):
    rip_current_detector = "rip_current_detector"


class TorchvisionModelVersion(str, Enum):
    one = "1"
