import os
import errno
import logging
# import sys
# import warnings

from pathlib import Path
from typing import Any
# from urllib.request import urlopen
from datetime import datetime, timezone
# import cupy as cp
# import matplotlib.pyplot as plt
# import requests
import torch
import torchvision
from fastapi import Depends, FastAPI, UploadFile, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette_exporter import PrometheusMiddleware, handle_metrics
from starlette_exporter.optional_metrics import request_body_size, response_body_size
# from PIL import Image
from pydantic import BaseModel
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from model_version import TorchvisionModelName, TorchvisionModelVersion
from namify import namify_for_content
from score import ClassificationModelResult
from torchvision_processing import torchvision_process_image, get_device

logging.basicConfig( level=logging.INFO)


TORCHVISION_ENDPOINT_PREFIX = "/torchvision"
ALLOWED_IMAGE_EXTENSIONS = (
    "jpg",
    "png"
)

app = FastAPI()

app.add_middleware(
    PrometheusMiddleware,
    app_name='rip_current_detection',
    prefix='http',
    buckets=[0.1, 0.5, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    skip_paths=['/health', '/metrics', '/outputs', '/favicon.ico'],
    group_paths=False,
    optional_metrics=[response_body_size, request_body_size]
)
app.add_route("/metrics", handle_metrics)

# Prometheus metrics
# metrics_app = make_metrics_app()
# app.mount("/metrics", metrics_app)


@app.get("/", include_in_schema=False)
async def index():
    """Convenience redirect to OpenAPI spec UI for service."""
    return RedirectResponse("/docs")


class UrlParams(BaseModel):
    url: str


output_path = Path(os.environ.get(
    'OUTPUT_DIRECTORY',
    str(Path(__file__).with_name('outputs') / 'fastapi')
))

model_folder = Path(os.environ.get(
    'MODEL_DIRECTORY',
    str(Path(__file__).parent / "models" )
))


# Mounting the 'static' output files for the app
app.mount(
    "/outputs",
    StaticFiles(directory=output_path),
    name="outputs"
)


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


models = {
    'rip_current_detector': {
        '1': load_model(
                model_folder / 'rip_current_detector' / '1' / 'fasterrcnn_resnet50_fpn.pt'
            ),
    }
}


def get_model(model: str, version: str):
    return models[model][version]


def annotation_image_and_classification_result(
    url: str,
    classification_result: ClassificationModelResult
):

    dt = datetime.utcnow().replace( tzinfo=timezone.utc )
    dt_str = dt.isoformat( "T", "seconds" ).replace( '+00:00', 'Z' )

    return {
        "time": dt_str,
        "annotated_image_url": url,
        "classification_result": classification_result
    }


@app.post(
    f"{TORCHVISION_ENDPOINT_PREFIX}/{{model}}/{{version}}/upload",
    tags=['torchvision'],
    summary="Torchvision model prediection on image upload"
)
def torchvision_from_upload(
    request: Request,
    model: TorchvisionModelName,
    version: TorchvisionModelVersion,
    file: UploadFile,
    pt: Any = Depends(get_model),
):
    bytedata = file.file.read()

    ( name, ext ) = namify_for_content( bytedata )

    assert ext in ALLOWED_IMAGE_EXTENSIONS, \
        f"{ext} not in allowed image file types: {repr(ALLOWED_IMAGE_EXTENSIONS)}"

    ( res_path, classification_result ) = torchvision_process_image(
        pt,
        output_path,
        model.value,
        version.value,
        name,
        bytedata
    )

    if( res_path is None ):
        return annotation_image_and_classification_result(
            None,
            classification_result
        )

    rel_path = os.path.relpath( res_path, output_path )

    url_path_for_output = rel_path

    try:
        # Try for an absolute URL (prefixed with http(s)://hostname, etc.)
        url_path_for_output = str( request.url_for( 'outputs', path=rel_path ) )
    except Exception:
        # Fall back to the relative URL determined by the router
        url_path_for_output = app.url_path_for(
            'outputs', path=rel_path
        )
    finally:
        pass

    return annotation_image_and_classification_result(
        url_path_for_output,
        classification_result
    )


@app.post("/health")
def health():
    return { "health": "ok" }

# @app.post("/{model}/{version}/url")
# def from_url(
#     model: str,
#     version: str,
#     params: UrlParams,
#     pt: Any = Depends(get_model),
# ):
#     bytedata = requests.get(params.url).content
#     name = Path(params.url).name
#     proc = process_image(pt, model, version, name, bytedata)
#     return { "path": proc }
