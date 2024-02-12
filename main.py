from ultralytics import YOLO


def yolo8_test():
    model = YOLO('yolov8n.yml')
    results = model.track(source="https://youtu.be/LNw0DJXcvt4", show=True)
    print(results)


if __name__ == "__main__":
    yolo8_test()
