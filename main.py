from ultralytics import YOLO


def main():
    # 加载模型
    model = YOLO("./ultralytics/models/v8/yolov8n-DC.yaml")  # 从头开始构建新模型
    # model = YOLO("./runs/detect/train_DBB_all_500/weights/best.pt")  # 加载预训练模型（建议用于训练）

    # 使用模型
    model.train(data="./ultralytics/datasets/RUOD.yaml",
                epochs=400, device='cuda:0',
                batch=4,
                save_period=50,
                verbose=True,
                project="test",
                name="train_n_400",
                profile=True,)  # 训练模型
    metrics = model.val(name="val_n_400")  # 在验证集上评估模型性能
    # results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
    # results = model.predict(source='C://Users//xu//Desktop//vi//img',
    #                         save=True,
    #                         profile=True,
    #                         save_txt=True)
    # print(results.save_dir)
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式

    # model = YOLO("./models/v8/yolov8n.yaml")  # 从头开始构建新模型
    # model = YOLO("./weight_all/best_DCN_all_400.pt")  # 加载预训练模型（建议用于训练）
    #
    # # 使用模型
    # model.train(data="./datasets/fish_01.yaml",
    #             epochs=200, device='cuda:0',
    #             batch=32,
    #             save_period=50,
    #             verbose=True,
    #             project="exp_fish",
    #             name="train_DCN_all_fish_n_200",
    #             profile=True,
    #             close_mosaic=10)  # 训练模型
    # metrics = model.val(name="val_DCN_all_fish_n_200")  # 在验证集上评估模型性能


if __name__ == '__main__':
    main()
