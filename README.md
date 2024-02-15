# rip_current_detection

## Download model weights

```shell
# From rip_current_detection repo root
RCD_DIR="models/rip_current_detector/1" && \
    mkdir -p "${RCD_DIR}" && \
    wget https://www.dropbox.com/s/dcsdi36jbc570u9/fasterrcnn_resnet50_fpn.pt -O "${RCD_DIR}/fasterrcnn_resnet50_fpn.pt";
```

## Set up environment

### GPU setup on host

TODO

### Python environment setup


```shell
micromamba create -f environment.yml
micromamba activate rip_current_detection
```



## Run app

```shell
uvicorn api:app --port 8888 --reload
```
