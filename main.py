from ultralytics import YOLO


def main():
    # 加载模型
    model = YOLO("./ultralytics/models/v8/yolov8n-DC.yaml")  # 从头开始构建新模型

    # 使用模型
    model.train(data="./ultralytics/datasets/RUOD.yaml",
                epochs=400, device='cuda:0',
                batch=4,
                save_period=50,
                verbose=True,
                project="test",
                name="train_n_400",
                profile=True,)  # 训练模型
    metrics = model.val(name="val_n_400")  # 在验证集上评估模型性能.

if __name__ == '__main__':
    main()
