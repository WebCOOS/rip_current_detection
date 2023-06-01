import os
import sys
import warnings

import cupy as cp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from fastapi import FastAPI, Depends, UploadFile
from pydantic import BaseModel
from typing import Any
import requests
from pathlib import Path

# from google.colab.patches import cv2_imshow
warnings.filterwarnings('ignore')

# load the RIP dataset category names
RIP_INSTANCE_CATEGORY_NAMES = ['__background__', 'rip']


app = FastAPI()


class UrlParams(BaseModel):
    url: str


output_path = Path(os.environ.get(
    'OUTPUT_DIRECTORY',
    str(Path(__file__).with_name('outputs') / 'fastapi')
))

model_folder = Path(os.environ.get(
    'MODEL_DIRECTORY',
    str(Path(__file__).parent)
))

def load_model(weights_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=2
    )
    model.load_state_dict(torch.load(weights_path))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    return model

models = {
    'rip_current_detector': {
        '1': load_model(str(model_folder / 'rip_current_detector' / '1' / 'saved_weights.pt')),
    }
}

def get_model(model: str, version: str):
    return models[model][version]

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def get_prediction(img_path, model, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.

    """
    # img = Image.open(img_path)
    img = img_path
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img.to(get_device(), dtype=torch.float)])
    pred_class = [RIP_INSTANCE_CATEGORY_NAMES[i]
                  for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    print("pred_score ", pred_score)
    pred_t = [pred_score.index(x) for x in pred_score if x > confidence]
    if len(pred_t) == 0:
        pred_boxes = []
        pred_class = []
        pred_score = []
    else:
        pred_t = [pred_score.index(x)
                  for x in pred_score if x > confidence][-1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        pred_score = pred_score[:pred_t+1]

    return pred_boxes, pred_class, pred_score


def detect_object(img_path, model, confidence=0.5, rect_th=2, text_size=2, text_th=2):
    """
    object_detection_api
      parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
        - rect_th - thickness of bounding box
        - text_size - size of the class label text
        - text_th - thichness of the text
      method:
        - prediction is obtained from get_prediction method
        - for each prediction, bounding box is drawn and text is written 
          with opencv
        - the final image is displayed
    """
    boxes, pred_cls, pred_score = get_prediction(img_path, model, confidence)

    img = img_path

    # convert PIL image to OPENCV image
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return pred_cls, pred_score, boxes, img


def draw_prediction(img, boxes, pred_cls, rect_th=2, text_size=2, text_th=2):
    for i in range(len(boxes)):
        pt1 = (int(boxes[i][0][0]), int(boxes[i][0][1]))
        pt2 = (int(boxes[i][1][0]), int(boxes[i][1][1]))
        cv2.rectangle(img, pt1, pt2, color=(255, 0, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (255, 0, 0), thickness=text_th)

    if len(boxes) == 0:
        cv2.putText(img, 'No rip currents are detected', [
                    100, 100], cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 0, 0), thickness=text_th)

    return img


def process_image(pt_model, model: str, version: str, name: str, bytedata: bytes):
    npdata = np.asarray(bytearray(bytedata), dtype="uint8")
    image = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cls, score, bboxes, image = detect_object(image, pt_model, confidence=0.7)

    if bboxes:

        output_file = output_path / model / str(version) / name
        image = draw_prediction(image, bboxes, cls)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), image)
        return str(output_file)

    return None


@app.post("/{model}/{version}/upload")
def from_upload(
    model: str,
    version: str,
    file: UploadFile,
    pt: Any = Depends(get_model),
):
    bytedata = file.file.read()
    proc = process_image(pt, model, version, file.filename, bytedata)
    return { "path": proc }


@app.post("/{model}/{version}/url")
def from_url(
    model: str,
    version: str,
    params: UrlParams,
    pt: Any = Depends(get_model),
):
    bytedata = requests.get(params.url).content
    name = Path(params.url).name
    proc = process_image(pt, model, version, name, bytedata)
    return { "path": proc }


@app.post("/health")
def health():
    return { "health": "ok" }
