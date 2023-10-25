from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Training
results = model.train(
    data='data.yaml',
    imgsz=416,
    epochs=30,  
    batch=8,
    name='end_model'
)
