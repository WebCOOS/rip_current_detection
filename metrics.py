
import os
from prometheus_client import (
    make_asgi_app,
    CollectorRegistry,
    multiprocess,
    Counter,
    Histogram
)
from model_version import (
    ModelFramework,
    TorchvisionModelName,
    TorchvisionModelVersion,
)


MODEL_PREDICTION_TIMING_HISTOGRAM = Histogram(
    'model_prediction_timing_histogram',
    'Histogram',
    labelnames = [
        'model_framework',
        'model_name',
        'model_version',
    ],
    buckets = (
        0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 10, float("inf")
    )
)

OBJECT_CLASSIFICATION_DETECTION_COUNTER = Counter(
    'object_classification_detection_counter',
    'Overall count of inputs with successful detections (that meet a threshold)',
    [
        'model_framework',
        'model_name',
        'model_version',
        'classification_name',
    ]
)

OBJECT_CLASSIFICATION_OBJECT_COUNTER = Counter(
    'object_classification_object_counter',
    'Count of detected objects in all inputs (that meet a threshold)',
    [
        'model_framework',
        'model_name',
        'model_version',
        'classification_name',
    ]
)

# Per: <https://prometheus.github.io/client_python/instrumenting/labels/>
#   Metrics with labels are not initialized when declared, because the client
#   can’t know what values the label can have. It is recommended to initialize
#   the label values by calling the .labels() method alone:
#
#       c.labels('get', '/')

LABELS = (
    ( ModelFramework.TORCHVISION, TorchvisionModelName.rip_current_detector, TorchvisionModelVersion.one ),
)
__RIP_CURRENT = 'rip_current'

for ( fw, mdl, ver ) in LABELS:
    OBJECT_CLASSIFICATION_DETECTION_COUNTER.labels(
        fw.name,
        mdl.value,
        ver.value,
        __RIP_CURRENT,
    )

    OBJECT_CLASSIFICATION_OBJECT_COUNTER.labels(
        fw.name,
        mdl.value,
        ver.value,
        __RIP_CURRENT,
    )

ENV_PROMETHEUS_MULTIPROC_DIR = 'PROMETHEUS_MULTIPROC_DIR'


# def make_metrics_app():

#     if os.getenv( ENV_PROMETHEUS_MULTIPROC_DIR ) is None:
#         # TODO: This is a hack, in place to prevent the metrics wiring to die
#         # if we don't define its precious environment variable.
#         os.environ[ENV_PROMETHEUS_MULTIPROC_DIR] = "/tmp"

#     registry = CollectorRegistry()
#     multiprocess.MultiProcessCollector( registry )
#     return make_asgi_app( registry = registry )


def increment_rip_current_detection_counter(
    fw: str,
    mdl_name: str,
    mdl_version: str
):
    OBJECT_CLASSIFICATION_DETECTION_COUNTER.labels(
        fw,
        mdl_name,
        mdl_version,
        __RIP_CURRENT
    ).inc()

def increment_rip_current_object_counter(
    fw: str,
    mdl_name: str,
    mdl_version: str
):
    OBJECT_CLASSIFICATION_OBJECT_COUNTER.labels(
        fw,
        mdl_name,
        mdl_version,
        __RIP_CURRENT
    ).inc()

def time_model_prediction_context(
    fw: str,
    mdl_name: str,
    mdl_version: str
):
    return MODEL_PREDICTION_TIMING_HISTOGRAM.labels(
        fw,
        mdl_name,
        mdl_version,
    ).time()
