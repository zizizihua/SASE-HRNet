#!/usr/bin/env bash
T=`date +%m%d%H%M`

PROJECT=sase_hrnet_coco
WEIGHTS=runs/train/${PROJECT}/weights/best.pt
DATA=data/coco.yaml
IMG_SIZE=320
BATCH_SIZE=32
WORKERS=8
WORK_DIR=runs/val
DEVICE=$1
PY_ARGS=${@:2}

mkdir -p ${WORK_DIR}/${PROJECT}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -u val.py --weights ${WEIGHTS} \
    --data ${DATA} \
    --batch-size ${BATCH_SIZE} \
    --imgsz ${IMG_SIZE} \
    --workers ${WORKERS} \
    --name ${PROJECT} \
    --device ${DEVICE} \
    --half \
    --exist-ok \
    --coco_eval \
    ${PY_ARGS} \
    2>&1 | tee ${WORK_DIR}/${PROJECT}/val.$T.log
