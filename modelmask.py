from ultralytics import YOLO

model = YOLO("masked.pt")

# Run inference on a video
results = model("video2.mp4", show=True, save=True, save_dir="runs/detect/my_output")
