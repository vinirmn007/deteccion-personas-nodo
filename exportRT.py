from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("models/model_v9s.pt")

# Export the model
# model.export(format="engine")
model.export(format="engine", half=True, dynamic=True, imgsz=640)

# Load the exported TensorRT model
trt_model = YOLO("mo0del_v9n.engine")


