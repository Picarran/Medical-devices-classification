from ultralytics.models import YOLO

if __name__ == '__main__':
    model = YOLO(model='./runs/train/exp4/weights/best.pt')
    results = model.predict(source='./datasets/images/test/a_14_1_6_12_16.jpg', device='0', imgsz=640, project='./runs/detect/', name='exp')
    for result in results:
        print(result.boxes.data)
        result.show()
        result.save(filename="./runs/detect/result.jpg")
