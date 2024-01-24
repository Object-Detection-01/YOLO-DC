# YOLO-DC: YOLO-Object Detectors Based on Deformable Convolutions
<img src="images/compare.png" width="1000" >

## Introduction
YOLO-DC outperforms numerous state-of-the-art (SOTA) algorithms, including YOLOv8, while maintaining a comparable level of computation and parameter count.
For more details, please refer to our [report on Arxiv](https://arxiv.org/abs/2107.08430).

<img src="images/coco-compar-nsm-Para.png" width="1000" >
In the figure above, (a) and (b) depict comparisons of computational and parameter counts among the models on the COCO 2017 dataset, respectively.

## Benchmark

|Model |size |AP<sup>val<br>0.5:0.95 |AP<sup>val<br>0.5 | Params<br>(M) |FLOPs<br>(G)|
| ------        |:---: | :---:    | :---:       |:---:  | :---: |
|YOLO-DC-N   |640  |40.8 |56.9     |3.9 | 8.9 |
|YOLO-DC-S   |640  |46.6 |63.5     |13.9 | 29.2 |
|YOLO-DC-M   |640  |**50.4** |**67.3**     |32.9 | 70.9 |

Table Notes

- Results of the AP and speed are evaluated on [COCO val2017](https://cocodataset.org/#download) dataset with the input
  resolution of 640×640.
- All experiments are based on NVIDIA 3090 GPU.

## Environment

- python requirements

  ```shell
  pip install -r requirements.txt
  ```

- data:

  prepare [COCO](http://cocodataset.org)
  dataset, [YOLO format coco labels](https://github.com/meituan/YOLOv6/releases/download/0.1.0/coco2017labels.zip) and
  specify dataset paths in data.yaml

## Train

#### Gold-YOLO-N

- Step 1: Training a base model

  Be sure to open use_dfl mode in config file (use_dfl=True, reg_max=16)

  ```shell
  python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
  									--batch 128 \
  									--conf configs/gold_yolo-n.py \
  									--data data/coco.yaml \
  									--epoch 300 \
  									--fuse_ab \
  									--use_syncbn \
  									--device 0,1,2,3,4,5,6,7 \
  									--name gold_yolo-n
  ```

- Step 2: Self-distillation training

  Be sure to open use_dfl mode in config file (use_dfl=True, reg_max=16)

  ```shell
  python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
  									--batch 128 \
  									--conf configs/gold_yolo-n.py \
  									--data data/coco.yaml \
  									--epoch 300 \
  									--device 0,1,2,3,4,5,6,7 \
  									--use_syncbn \
  									--distill \
  									--teacher_model_path runs/train/gold_yolo_n/weights/best_ckpt.pt \
  									--name gold_yolo-n
  ```

#### Gold-YOLO-S/M/L

- Step 1: Training a base model

  Be sure to open use_dfl mode in config file (use_dfl=True, reg_max=16)

  ```shell
  python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
  									--batch 256 \
  									--conf configs/gold_yolo-s.py \ # gold_yolo-m/gold_yolo-l
  									--data data/coco.yaml \
  									--epoch 300 \
  									--fuse_ab \
  									--use_syncbn \
  									--device 0,1,2,3,4,5,6,7 \
  									--name gold_yolo-s # gold_yolo-m/gold_yolo-l
  ```

- Step 2: Self-distillation training

  Be sure to open use_dfl mode in config file (use_dfl=True, reg_max=16)

  ```shell
  python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
  									--batch 256 \ # 128 for distillation of gold_yolo-l
  									--conf configs/gold_yolo-s.py \ # gold_yolo-m/gold_yolo-l
  									--data data/coco.yaml \
  									--epoch 300 \
  									--device 0,1,2,3,4,5,6,7 \
  									--use_syncbn \
  									--distill \
  									--teacher_model_path runs/train/gold_yolo-s/weights/best_ckpt.pt \
  									--name gold_yolo-s # gold_yolo-m/gold_yolo-l
  ```

## Evaluation

```shell
python tools/eval.py --data data/coco.yaml --batch 32 --weights weights/Gold_s_pre_dist.pt --task val --reproduce_640_eval
```


## Acknowledgement

The implementation is based on [YOLOv8](https://github.com/ultralytics/ultralytics). Thanks for their open source code.
