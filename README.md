# rip_current_detection

## Download model weights

```sh
# From rip_current_detection repo root
mkdir models
wget https://www.dropbox.com/s/dcsdi36jbc570u9/fasterrcnn_resnet50_fpn.pt -O models/fasterrcnn_resnet50_fpn.pt
```

## Set up environment

```sh
conda create -n rip_det python=3.9
conda activate rip_det
pip install -r requirements.txt
```

## Run app

```sh
uvicorn api:app --reload
```
