from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")

# Use the model
model.train(data="dataset/dataset.yaml", epochs=100, imgsz=640, batch=16)  # train the model
metrics = model.val()
path = model.export(format="onnx")
