from ultralytics import YOLO


def yolo8_test():
    model = YOLO('yolov8n.pt')
    results = model.train(data='home/mkokhaie/PycharmProjects/Yolo8/datasets/dataset/data.yml', epochs=100, imgsz=640)
    print(results)


if __name__ == "__main__":
    yolo8_test()
