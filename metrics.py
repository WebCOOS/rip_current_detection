
import os
from prometheus_client import (
    make_asgi_app,
    CollectorRegistry,
    multiprocess,
    Counter
)
from model_version import (
    ModelFramework,
    TorchvisionModelName,
    TorchvisionModelVersion,
)


OBJECT_CLASSIFICATION_COUNTER = Counter(
    'object_classification_counter',
    'Count of classifications',
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
    OBJECT_CLASSIFICATION_COUNTER.labels(
        fw.name,
        mdl.value,
        ver.value,
        __RIP_CURRENT,
    )

ENV_PROMETHEUS_MULTIPROC_DIR='PROMETHEUS_MULTIPROC_DIR'

def make_metrics_app():

    if os.getenv( ENV_PROMETHEUS_MULTIPROC_DIR ) is None:
        # TODO: This is a hack, in place to prevent the metrics wiring to die
        # if we don't define its precious environment variable.
        os.environ[ENV_PROMETHEUS_MULTIPROC_DIR] = "/tmp"

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector( registry )
    return make_asgi_app( registry = registry )


def increment_rip_current_counter(
    fw: str,
    mdl_name: str,
    mdl_version: str
):
    OBJECT_CLASSIFICATION_COUNTER.labels(
        fw,
        mdl_name,
        mdl_version,
        __RIP_CURRENT
    ).inc()
