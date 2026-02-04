from ultralytics import YOLO

model = YOLO("cigarettes2.pt")

model.predict(source=1, show=True)
