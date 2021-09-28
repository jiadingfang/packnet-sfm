# Handy commands:
# - `make docker-build`: builds DOCKERIMAGE (default: `packnet-sfm:latest`)
# PROJECT ?= packnet-sfm-ucm-128x128-kitti
# PROJECT ?= packnet-sfm-ucm-192x640-kitti
# PROJECT ?= packnet-sfm-intrinsic-128x128-kitti
# PROJECT ?= packnet-sfm-intrinsic-192x640-kitti
# PROJECT ?= packnet-sfm-ucm-128x128-omnicam
# PROJECT ?= packnet-sfm-ucm-384x384-omnicam
# PROJECT ?= packnet-sfm-ucm-512x512-omnicam
# PROJECT ?= packnet-sfm-ucm-1024x1024-omnicam
# PROJECT ?= packnet-sfm-ucm-384x640-ddad
# PROJECT ?= packnet-sfm-intrinsic-128x128-omnicam
# PROJECT ?= packnet-sfm-intrinsic-192x640-omnicam
# PROJECT ?= packnet-sfm-ucm-752x480-euroc
# PROJECT ?= packnet-sfm-ucm-256x384-euroc
PROJECT ?= packnet-infer
# PROJECT ?= packnet-test
# PROJECT ?= packnet-test1
# PROJECT ?= packnet-test2
# PROJECT ?= packnet-test3
# PROJECT ?= packnet-train
# PROJECT ?= packnet-double
# PROJECT ?= packnet-double1
# PROJECT ?= packnet-double2
# PROJECT ?= packnet-double3
# PROJECT ?= packnet-ds
# PROJECT ?= packnet-fov
# PROJECT ?= packnet-pd


WORKSPACE ?= /workspace/$(PROJECT)
DOCKER_IMAGE ?= ${PROJECT}:latest

SHMSIZE ?= 444G
WANDB_MODE ?= run
WANDB_API_KEY=0deb7a9c0b27618e1de0b5c1468716fbc482f1a1
DOCKER_OPTS := \
			--name ${PROJECT} \
			--rm -it \
			--shm-size=${SHMSIZE} \
			-e AWS_DEFAULT_REGION \
			-e AWS_ACCESS_KEY_ID \
			-e AWS_SECRET_ACCESS_KEY \
			-e WANDB_API_KEY=${WANDB_API_KEY} \
			-e WANDB_ENTITY \
			-e WANDB_MODE \
			-e HOST_HOSTNAME= \
			-e OMP_NUM_THREADS=1 -e KMP_AFFINITY="granularity=fine,compact,1,0" \
			-e OMPI_ALLOW_RUN_AS_ROOT=1 \
			-e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
			-e NCCL_DEBUG=VERSION \
            -e DISPLAY=${DISPLAY} \
            -e XAUTHORITY \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
			-v ~/.aws:/root/.aws \
			-v /root/.ssh:/root/.ssh \
			-v ~/.cache:/root/.cache \
			-v /data:/data \
			-v /mnt/fsx/:/mnt/fsx \
			-v /dev/null:/dev/raw1394 \
			-v /tmp:/tmp \
			-v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0 \
			-v /var/run/docker.sock:/var/run/docker.sock \
			-v ${PWD}:${WORKSPACE} \
			-v /scratch/fjd:/data/datasets \
			-w ${WORKSPACE} \
			--privileged \
			--ipc=host \
			--network=host

NGPUS=$(shell nvidia-smi -L | wc -l)
MPI_CMD=mpirun \
		-allow-run-as-root \
		-np ${NGPUS} \
		-H localhost:${NGPUS} \
		-x MASTER_ADDR=127.0.0.1 \
		-x MASTER_PORT=23457 \
		-x HOROVOD_TIMELINE \
		-x OMP_NUM_THREADS=1 \
		-x KMP_AFFINITY='granularity=fine,compact,1,0' \
		-bind-to none -map-by slot -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=4 \
		--report-bindings


.PHONY: all clean docker-build docker-overfit-pose

all: clean

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

docker-build:
	docker build \
		-f docker/Dockerfile \
		-t ${DOCKER_IMAGE} .

docker-start-interactive: docker-build
	nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} bash

docker-start-jupyter: docker-build
	nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
		bash -c "jupyter notebook --port=8888 -ip=0.0.0.0 --allow-root --no-browser"

docker-run: docker-build
	nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
		bash -c "${COMMAND}"

docker-run-mpi: docker-build
	nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
		bash -c "${MPI_CMD} ${COMMAND}"