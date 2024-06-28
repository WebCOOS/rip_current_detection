import os
from typing import Union
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from score import ClassificationModelResult, BoundingBoxPoint
from model_version import (
    ModelFramework,
    YOLOModelName,
    YOLOModelVersion
)
from metrics import (
    increment_rip_current_detection_counter,
    increment_rip_current_object_counter,
    time_model_prediction_context
)
import logging
import torch

logger = logging.getLogger( __name__ )
width = 896
height = 896
DEFAULT_YOLO_THRESHOLD = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX

MODEL_FOLDER = Path(os.environ.get(
    "MODEL_DIRECTORY",
    str(Path(__file__).parent)
))

YOLO_MODELS = {
    # public-facing model name
    "ripdetect_walton": {
        # public-facing model version
        "yolov8x_1.1": YOLO(
            str(
                MODEL_FOLDER \
                / "ripdetect_walton" \
                / "yolov8x_1.1" \
                / "yolov8x_1.1.pt"
            )
        ),
    }
}

__THE_DEVICE=None


def get_device():
    """Gets device, preferring a CUDA-enabled GPU when available"""

    global __THE_DEVICE

    if __THE_DEVICE is not None:
        return __THE_DEVICE

    if torch.cuda.is_available():
        logger.warning( "cuda device enabled for ultralytics!")
        __THE_DEVICE = torch.device('cuda')
    else:
        logger.warning( "cuda unavailable for ultralytics, cpu device used instead")
        __THE_DEVICE = torch.device('cpu')

    return __THE_DEVICE


def get_yolo_model(model: YOLOModelName, version: YOLOModelVersion):
    return YOLO_MODELS[model.value][version.value]


def yolo_process_image(
    yolo_model: YOLO,
    output_path: Path,
    model: Union[YOLOModelName, str],
    version: Union[YOLOModelVersion, str],
    name: str,
    bytedata: bytes,
    confidence_threshold: float = None,
):

    assert yolo_model, \
        f"Must have yolo_model passed to {yolo_process_image.__name__}"

    assert output_path and isinstance( output_path, Path ), \
        f"output_path parameter for {yolo_process_image.__name__} is not Path"

    assert output_path.exists() and output_path.is_dir(), \
        (
            f"output_path parameter for {yolo_process_image.__name__} must exist "
            "and be a directory"
        )

    assert isinstance( model, ( YOLOModelName, str ) )
    assert isinstance( version, ( YOLOModelVersion, str ) )

    if( isinstance( model, YOLOModelName ) ):
        model = model.value

    if( isinstance( version, YOLOModelVersion ) ):
        version = version.value

    if confidence_threshold is None:
        confidence_threshold = DEFAULT_YOLO_THRESHOLD

    ret: ClassificationModelResult = ClassificationModelResult(
        ModelFramework.ULTRALYTICS.value,
        model,
        version
    )

    output_file = output_path / model / str(version) / name

    npdata = np.asarray(bytearray(bytedata), dtype="uint8")
    frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_boxes = frame

    results = None

    labels: list = [
        ModelFramework.ULTRALYTICS.value,
        model,
        version
    ]

    with time_model_prediction_context( *labels ):

        #use YOLOv8
        results = yolo_model.predict(
            frame,
            # conf = 0.1,
            device = get_device()
        )

    # If any score is above threshold, flag it as detected
    detected = False

    for result in results:
        #for score, cls, cls_name, bbox in zip(result.boxes.conf, result.boxes.cls, result.names, result.boxes.xyxy):
        for box in result.boxes:

            score = box.conf.item()
            cls = int(box.cls.item())
            cls_name = yolo_model.names[cls]

            if score < confidence_threshold:
                continue

            detected = True

            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            h, w, _ = frame.shape

            y_min = int(max(1, y1))
            x_min = int(max(1, x1))
            y_max = int(min(h, y2))
            x_max = int(min(w, x2))

            if detected is True:

                label = cls_name + ": " + ": {:.2f}%".format(score * 100)
                img_boxes = cv2.rectangle(img_boxes, (x_min, y_max), (x_max, y_min), (0, 0, 255), 2)
                cv2.putText(img_boxes, label, (x_min, y_max - 10), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                ret.add(
                    classification_name=cls_name,
                    classification_score=score,
                    bbox=(
                        BoundingBoxPoint( x_min, y_min ),
                        BoundingBoxPoint( x_max, y_max ),
                    )
                )

                # Update object metrics
                increment_rip_current_object_counter(
                    ModelFramework.ULTRALYTICS.name,
                    model,
                    version
                )

    # outp = cv2.resize(img_boxes, (1280, 720))

    if detected is True:

        # Update Prometheus metrics for each of the classes that were
        # detected
        increment_rip_current_detection_counter(
            ModelFramework.ULTRALYTICS.name,
            model,
            version,
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), img_boxes )
        return ( str(output_file), ret )

    return ( None, ret  )
