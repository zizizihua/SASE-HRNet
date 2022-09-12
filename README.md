# SASE-HRNet
Scale-Aware Squeeze-and-Excitation for Lightweight Object Detection. [[paper]()]


## Performance on COCO2017 val
| Model | Input Size | Params(M) | FLOPs(G) | mAP |
| --- | --: | --: | --: | --: |
| [SASE-HRNet-S](https://github.com/zizizihua/SASE-HRNet/releases/download/v1.0.0/sase_hrnet_s.pt) | 320 | 0.8 | 0.6 | 23.6 |
| [SASE-HRNet-L](https://github.com/zizizihua/SASE-HRNet/releases/download/v1.0.0/sase_hrnet_l.pt) | 320 | 1.2 | 0.9 | 27.7 |


## Requirements
+ pip install -r requirements.txt
+ Download COCO/VOC dataset, and convert dataset to YOLOv5 format


## Testing
```shell
python -u val.py \
    --weights ${WEIGHTS} \
    --data data/coco.yaml \
    --batch-size 32 \
    --imgsz 320 \
    --workers 8 \
    --device 0 \
    --half \
    --coco_eval
```

## Training
```shell
python -u train.py \
    --cfg models/sase-hrnet-s \
    --data data/coco.yaml \
    --hyp data/hyps/hyp.scratch-low.yaml \
    --epochs 300 \
    --batch-size 64 \
    --imgsz 416 \
    --workers 8 \
    --device 0 \
    --fp16
```

## Acknowledgement
This code is based on the [yolov5](https://github.com/glenn-jocher/yolov5) framework.
Thank [@glenn-jocher](https://github.com/glenn-jocher) for his excellent work.