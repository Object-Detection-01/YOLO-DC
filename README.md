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
  If you are prompted that a package is missing, follow the corresponding prompts to follow that package.
- data:

  prepare [COCO](http://cocodataset.org)
  dataset, YOLO format coco labels and
  specify dataset paths in data.yaml (data.yaml is located at ". /ultralytics/datasets/coco.yaml").

## Train

  ### 1. command-line mode
  - See train.py for more information on how to use it.

  ```shell
  python ./train.py 
                --yaml ultralytics/models/v8/yolov8n.yaml
                --conf configs/gold_yolo-n.py \
                --data data/coco.yaml \
                --epoch 300 \
                --fuse_ab \
                --use_syncbn \
                --device 0,1,2,3,4,5,6,7 \
                --name gold_yolo-n
  ```
  ### 2. python
  


## Evaluation

```shell
python tools/eval.py --data data/coco.yaml --batch 32 --weights weights/Gold_s_pre_dist.pt --task val --reproduce_640_eval
```


## Acknowledgement

The implementation is based on [YOLOv8](https://github.com/ultralytics/ultralytics). Thanks for their open source code.
