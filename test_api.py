import requests

json_response = requests.post(
    'http://127.0.0.1:8888/rip_current_detector/1/url',
    headers={
        'content-type': 'application/json'
    },
    json={
        'url': 'https://files.axds.co/webcoos/rip_current_detection/inputs/rip-255.jpg',
    }
)

# Shouldn't produce an output file
json_response = requests.post(
    'http://127.0.0.1:8888/rip_current_detector/1/url',
    headers={
        'content-type': 'application/json'
    },
    json={
        'url': 'https://files.axds.co/webcoos/rip_current_detection/inputs/norip-120.jpg',
    }
)

with open('inputs/rip-45.jpg', 'rb') as f:
    json_response = requests.post(
        'http://127.0.0.1:8888/rip_current_detector/1/upload',
        files={
            'file': ('rip-45.jpg', f)
        }
    )

# Shouldn't produce an output file
with open('inputs/norip-99.png', 'rb') as f:
    json_response = requests.post(
        'http://127.0.0.1:8888/rip_current_detector/1/upload',
        files={
            'file': ('norip-99.png', f)
        }
    )