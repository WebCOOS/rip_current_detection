import os
import logging
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from fastapi import Depends, FastAPI, UploadFile, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette_exporter import PrometheusMiddleware, handle_metrics
from starlette_exporter.optional_metrics import request_body_size, response_body_size
from pydantic import BaseModel
from model_version import (
    TorchvisionModelName,
    TorchvisionModelVersion,
    YOLOModelName,
    YOLOModelVersion
)
from namify import namify_for_content
from score import ClassificationModelResult
from torchvision_processing import torchvision_process_image, get_torchvision_model
from ultralytics_processing import yolo_process_image, get_yolo_model

logging.basicConfig( level=logging.INFO)


TORCHVISION_ENDPOINT_PREFIX = "/torchvision"
ULTRALYTICS_ENDPOINT_PREFIX = "/ultralytics"
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

# model_folder = Path(os.environ.get(
#     'MODEL_DIRECTORY',
#     str(Path(__file__).parent / "models" )
# ))


# Mounting the 'static' output files for the app
app.mount(
    "/outputs",
    StaticFiles(directory=output_path),
    name="outputs"
)


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
    pt: Any = Depends(get_torchvision_model),
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


@app.post(
    f"{ULTRALYTICS_ENDPOINT_PREFIX}/{{model}}/{{version}}/upload",
    tags=['ultralytics'],
    summary="Ultralytics/YOLO prediction on image upload"
)
def yolo_from_upload(
    request: Request,
    model: YOLOModelName,
    version: YOLOModelVersion,
    file: UploadFile,
    confidence_threshold: float = Form( gt=0.0, lt=1.0, default=None ),
    pt: Any = Depends(get_yolo_model),
):
    bytedata = file.file.read()

    ( name, ext ) = namify_for_content( bytedata )

    assert ext in ALLOWED_IMAGE_EXTENSIONS, \
        f"{ext} not in allowed image file types: {repr(ALLOWED_IMAGE_EXTENSIONS)}"

    ( res_path, classification_result ) = yolo_process_image(
        pt,
        output_path,
        model.value,
        version.value,
        name,
        bytedata,
        confidence_threshold
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
