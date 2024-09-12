from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="data_custom.yaml", imgsz=640, epochs=100, workers=1)