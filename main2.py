from ultralytics import YOLO

model = YOLO("cig\dotpt\cigarettes2.pt")

model.predict(source=1, show=True)
