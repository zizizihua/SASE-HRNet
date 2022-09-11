#!/usr/bin/env bash
T=`date +%m%d%H%M`

CONFIG=models/sase-hrnet-s.yaml
DATA=data/coco.yaml
HYP=data/hyps/hyp.scratch-low.yaml
EPOCHS=300
IMG_SIZE=416
BATCH_SIZE=64
CPU_NUM=8
WORK_DIR=runs/train
PROJECT=sase-hrnet-s-coco
DEVICE=$1
PY_ARGS=${@:2}

mkdir -p ${WORK_DIR}/${PROJECT}

export OMP_NUM_THREADS=$CPU_NUM
export OPENBLAS_NUM_THREADS=$CPU_NUM
export MKL_NUM_THREADS=$CPU_NUM
export VECLIB_MAXIMUM_THREADS=$CPU_NUM
export NUMEXPR_NUM_THREADS=$CPU_NUM

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u train.py --cfg ${CONFIG} \
    --data ${DATA} \
    --hyp ${HYP} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --imgsz ${IMG_SIZE} \
    --workers ${CPU_NUM} \
    --launcher "pytorch" \
    --name ${PROJECT} \
    --device ${DEVICE} \
    --fp16 \
    --exist-ok \
    ${PY_ARGS} \
    2>&1 | tee ${WORK_DIR}/${PROJECT}/train.$T.log
