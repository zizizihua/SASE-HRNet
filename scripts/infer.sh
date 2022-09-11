#!/usr/bin/env bash
T=`date +%m%d%H%M`

MODEL=$1  # sase_hrnet_voc
WEIGHTS=runs/train/${MODEL}/weights/best.pt
DATA=data/voc.yaml
SOURCE=$2
IMG_SIZE=320
WORK_DIR=runs/detect
PROJECT=${MODEL}
DEVICE=$3
PY_ARGS=${@:4}

mkdir -p ${WORK_DIR}/${PROJECT}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u detect.py --weights ${WEIGHTS} \
    --data ${DATA} \
    --source ${SOURCE} \
    --imgsz ${IMG_SIZE} \
    --name ${PROJECT} \
    --device ${DEVICE} \
    --half \
    --exist-ok \
    --conf-thres 0.25 \
    ${PY_ARGS}

#    --hide-labels \
