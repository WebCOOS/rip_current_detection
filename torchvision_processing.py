
import os
import errno
import logging
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path
from typing import Any, List, Tuple, Union
from model_version import ModelFramework, TorchvisionModelName, TorchvisionModelVersion
from score import BoundingBoxPoint, ClassificationModelResult
from metrics import (
    increment_rip_current_detection_counter,
    increment_rip_current_object_counter,
    time_model_prediction_context
)


logger = logging.getLogger( __name__ )


# Used to filter out low-confidence detections
THRESHOLD = 0.7

# Image annotation configuration
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (255, 0, 0)
BOX_THICKNESS = 2
TEXT_SIZE = 2
TEXT_THICKNESS = 2
RIP_TEXT = 'rip'

__THE_DEVICE=None


model_folder = Path(os.environ.get(
    'MODEL_DIRECTORY',
    str(Path(__file__).parent / "models" )
))


def get_device():
    """Gets device, preferring a CUDA-enabled GPU when available"""

    global __THE_DEVICE

    if __THE_DEVICE is not None:
        return __THE_DEVICE

    if torch.cuda.is_available():
        logger.warning( "cuda device enabled!")
        __THE_DEVICE = torch.device('cuda')
    else:
        logger.warning( "cuda unavailable, cpu device used instead")
        __THE_DEVICE = torch.device('cpu')

    return __THE_DEVICE


def load_model(model_path):
    """Loads model / model weights and moves to available device"""

    if not model_path.exists():
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror( errno.ENOENT ),
            str( model_path ),
        )

        # weights_path.parent.mkdir(parents=True, exist_ok=True)
        # r = requests.get(model_url)
        # r.raise_for_status()
        # with open(weights_path, 'wb') as f:
        #     f.write(r.content)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=2
    )

    the_device = get_device()
    model.load_state_dict(
        torch.load(
            str(model_path),
            map_location=the_device
        )
    )
    model.to(the_device)
    model.eval()

    # model.share_memory()

    return model


__TORCHVISION_MODELS: dict = None

def get_torchvision_model(model: TorchvisionModelName, version: TorchvisionModelVersion):

    global __TORCHVISION_MODELS

    if __TORCHVISION_MODELS is None:

        # load the models, cache in memory, and return
        __TORCHVISION_MODELS = {
            'rip_current_detector': {
                '1': load_model(
                        model_folder / 'rip_current_detector' / '1' / 'fasterrcnn_resnet50_fpn.pt'
                    ),
            }
        }

    return __TORCHVISION_MODELS[model.value][version.value]


def get_boxes(image, model, threshold) -> List[Tuple[float, Any]]:
    """Gets boxes of detected rip currents in image coordinates

    Only boxes whose corresponding scores are greater than the given threshold
    are returned.
    """
    with torch.no_grad():
        transform = T.Compose([T.ToTensor()])
        image = transform(image)
        predictions = model([image.to(get_device(), dtype=torch.float)])[0]
        boxes = []
        for box, score in zip(predictions['boxes'], predictions['scores']):
            if score > threshold:
                boxes.append( (float(score), box) )
        return boxes


def draw_boxes( image, boxes: List[Tuple[float, Any]] ):
    """Draws boxes on an image around detected rip currents"""

    for (score, box ) in boxes:
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        cv2.rectangle(
            image,
            pt1,
            pt2,
            color=COLOR,
            thickness=BOX_THICKNESS
        )

        score_percentage = round( 100 * score, 0 )

        cv2.putText(
            image,
            f"{RIP_TEXT}: {score_percentage}%",
            pt1,
            FONT,
            TEXT_SIZE,
            COLOR,
            thickness=TEXT_THICKNESS
        )

    return image


def torchvision_process_image(
    pt_model,
    output_path: Path,
    model: Union[TorchvisionModelName, str],
    version: Union[TorchvisionModelVersion, str],
    name: str,
    bytedata: bytes
) -> Tuple[str, ClassificationModelResult]:
    """Applies a given rip current detection model to the provided image"""

    assert pt_model, \
        f"Must have pt_model passed to {torchvision_process_image.__name__}"

    assert output_path and isinstance( output_path, Path ), \
        f"output_path parameter for {torchvision_process_image.__name__} is not Path"

    assert output_path.exists() and output_path.is_dir(), \
        (
            f"output_path parameter for {torchvision_process_image.__name__} must exist "
            "and be a directory"
        )

    assert isinstance( model, ( TorchvisionModelName, str ) )
    assert isinstance( version, ( TorchvisionModelVersion, str ) )

    if( isinstance( model, TorchvisionModelName ) ):
        model = model.value

    if( isinstance( version, TorchvisionModelVersion ) ):
        version = version.value

    ret: ClassificationModelResult = ClassificationModelResult(
        ModelFramework.TORCHVISION.value,
        model,
        version
    )

    npdata = np.asarray(bytearray(bytedata), dtype="uint8")
    input_image = cv2.imdecode(npdata, cv2.IMREAD_COLOR )
    # 'Plain' copy of input image, suitable for later applying any bounding
    # boxes for detections
    annotatable_image = cv2.imdecode(npdata, cv2.IMREAD_COLOR )

    # Convert the input image from BGR to RGB
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    boxes = None

    labels: list = [
        ModelFramework.ULTRALYTICS.value,
        model,
        version
    ]

    with time_model_prediction_context( *labels ):

        boxes = get_boxes( input_image, pt_model, THRESHOLD )


    detected = False

    if len( boxes ) > 0:
        # We've returned detected bounding boxes that meet with the threshold,
        # this is considered a detection
        detected = True

        for ( score, box ) in boxes:
            ret.add(
                classification_name = RIP_TEXT,
                classification_score = score,
                bbox = (
                    BoundingBoxPoint( int( box[0] ), int( box[1] ) ),
                    BoundingBoxPoint( int( box[2] ), int( box[3] ) ),
                )
            )

            # Increment metrics
            increment_rip_current_object_counter(
                ModelFramework.TORCHVISION.name,
                model,
                version
            )

    if detected:

        # Increment metrics
        increment_rip_current_detection_counter(
            ModelFramework.TORCHVISION.name,
            model,
            version
        )

        # Draw boxes on output to provide annotated image output
        output_file = output_path / model / str(version) / name

        # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        annotatable_image = draw_boxes( annotatable_image, boxes)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), annotatable_image)
        return (
            str(output_file),
            ret
        )

    return ( None, ret )
