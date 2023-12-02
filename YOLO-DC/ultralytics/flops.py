import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import RTDETRDetectionModel, DetectionModel

# model = RTDETRDetectionModel(cfg="E:/DL/ultralytics-xcz/ultralytics/models/v8/yolov8.yaml")  # build model
model = DetectionModel(cfg='E:/DL/ultralytics-xcz/ultralytics/models/v8/yolov8.yaml')  # build model
# model = YOLO("./models/v8/yolov8n_DCN.yaml")  # build model
# print(model)

input = torch.randn(1,3, 640, 640)
model.predict(input, profile=True)
