from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('best.pt')

# Define path to video file
source = 'DJI_0845.MP4'

# Run inference on the source
model.predict(source, save=True, conf=0.5)