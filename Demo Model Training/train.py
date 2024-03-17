from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(data = "soccer-demo-v2/data.yaml", batch = 32, epochs = 100)