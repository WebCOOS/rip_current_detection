# rip_current_detection

## Download model weights

```shell
# From rip_current_detection repo root
RCD_DIR="models/rip_current_detector/1" && \
    mkdir -p "${RCD_DIR}" && \
    wget https://www.dropbox.com/s/dcsdi36jbc570u9/fasterrcnn_resnet50_fpn.pt -O "${RCD_DIR}/fasterrcnn_resnet50_fpn.pt";
```

## Set up environment

### Python environment setup

```shell
micromamba create -f environment.yml
micromamba activate rip_current_detection
```

### GPU setup on host

This project can run against an NVIDIA CUDA-capable GPU, which will greatly
accelerate detection requests. You must have the following installed and
configured:

*   An NVIDIA GPU, capable of CUDA 11.x or 12.x, with up-to-date drivers
    installed for your target operating system.

*   If running within a Docker container, you will need the
    [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
    to be installed, and the `nvidia` container runtime configured as runtime
    within your Docker daemon.

If CUDA capabilities are detected at runtime, the API should detect and use the
device (using `torch.cuda.is_available('cuda')`). If not, the `cpu` device is
used as a fallback.

GPU hardware developed and tested on:

*   **OS**:                         Pop!_OS/Ubuntu 22.04  \
    **CPU**:                        Intel Xeon CPU E3-1245 v5 @ 3.50GHz  \
    **GPU**:                        Quadro P1000 (4GB VRAM)  \
    **Driver**:                     550.67  \
    **CUDA**:                       12.4  \
    **nvidia-container-toolkit**:   1.12.1


## Run app

Run on the host in 'development' mode (restarting on file changes):

```shell
uvicorn api:app --port 8888 --reload
```

To run in a Docker container (using the default `docker-compose.yml`):

```shell
docker compose up
```

To run a Docker container with the NVIDIA GPU capabilities assigned to the
container, you should use the `docker-compose.gpu.yml` Docker compose spec,
like so:

```shell
docker compose -f docker-compose.gpu.yml up
```
