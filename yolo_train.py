from ultralytics import YOLO


def main():

    model = YOLO('yolov8s.pt')
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=4,
        name='circle_detector',
        workers=0
    )



if __name__ == '__main__':
    main()
