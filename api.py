import os
import sys
import warnings
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import cupy as cp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision
import torchvision.transforms as T
from fastapi import Depends, FastAPI, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

# Used to filter out low-confidence detections
THRESHOLD = 0.7

# Image annotation configuration
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (255, 0, 0)
BOX_THICKNESS = 2
TEXT_SIZE = 2
TEXT_THICKNESS = 2
RIP_TEXT = 'rip'


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

def get_device():
    """Gets device, preferring a CUDA-enabled GPU when available"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def load_model(weights_path, model_url):
    """Loads model weights and moves to available device"""
    if not weights_path.exists():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(model_url)
        r.raise_for_status()
        with open(weights_path, 'wb') as f:
            f.write(r.content)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=2
    )
    model.load_state_dict(torch.load(str(weights_path)))

    model.to(get_device())
    model.eval()

    return model

models = {
    'rip_current_detector': {
        '1': load_model(
                model_folder / 'rip_current_detector' / '1' / 'saved_weights.pt',
                'https://www.dropbox.com/s/dcsdi36jbc570u9/fasterrcnn_resnet50_fpn.pt?dl=1'
            ),
    }
}

def get_model(model: str, version: str):
    return models[model][version]

def get_boxes(image, model, threshold):
    """Gets boxes of detected rip currents in image coordinates

    Only boxes whose corresponding scores are greater than the given threshold
    are returned.
    """
    transform = T.Compose([T.ToTensor()])
    image = transform(image)
    predictions = model([image.to(get_device(), dtype=torch.float)])[0]
    boxes = []
    for box, score in zip(predictions['boxes'], predictions['scores']):
        if score > threshold:
            boxes.append(box)
    return boxes

def draw_boxes(image, boxes):
    """Draws boxes on an image around detected rip currents"""
    for box in boxes:
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        cv2.rectangle(image, pt1, pt2, color=COLOR, thickness=BOX_THICKNESS)
        cv2.putText(image, RIP_TEXT, pt1, FONT, TEXT_SIZE, COLOR, thickness=TEXT_THICKNESS)

    return image

def process_image(pt_model, model: str, version: str, name: str, bytedata: bytes):
    """Applies a given rip current detection model to the provided image"""
    npdata = np.asarray(bytearray(bytedata), dtype="uint8")
    image = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = get_boxes(image, pt_model, THRESHOLD)

    if boxes:
        output_file = output_path / model / str(version) / name
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = draw_boxes(image, boxes)
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
