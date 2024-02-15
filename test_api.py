import requests
import logging
from pathlib import Path

logging.basicConfig( level=logging.INFO)
logger = logging.getLogger( __file__ )

BASE_DIR = Path( __file__ ).parent

logger.info( "Begin test battery...")

UPLOAD_TEST_PATHS = [
    ( 'rip-45.jpg',     BASE_DIR / 'inputs' / 'rip-45.jpg',     True    ),
    ( 'norip-99.png',   BASE_DIR / 'inputs' / 'norip-99.png',   False   ),
]

for ( name, up_path, is_detected ) in UPLOAD_TEST_PATHS:
    logger.info( f"Testing: {name} ('{str(up_path)}')")
    with open( up_path, 'rb') as f:
        json_response = requests.post(
            'http://localhost:8888/torchvision/rip_current_detector/1/upload',
            files={
                'file': ( name, f )
            }
        )

        result = json_response.json()

        assert result['classification_result']['detected'] == is_detected, \
            (
                f"Unexpected result from {name}: "
                f"{result['classification_result']['detected']} "
                f"!= {is_detected}"
            )

        logger.info(
            (
                f"Result for {name}: "
                f"{result['classification_result']['detected']}"
            )
        )

logger.info( "End test battery.")