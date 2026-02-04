from ultralytics import YOLO

model = YOLO("cigarettes2.pt") #change u
model.predict(source=1, show=True)
