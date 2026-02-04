import cv2
from ultralytics import YOLO

model = YOLO("cigarettes.pt")

colors = {
    0: (0,255,0),
    1: (255,0,0),
    2: (0,0,255)
}

labels = {
    0: "gak nyebat",
    1: "nyebat",
    2: "cig"
}

results = model.predict(source=1, stream=True)

for r in results:
    frame = r.orig_img

    for box in r.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        color = colors.get(cls, (255,255,255))
        label = labels.get(cls, "UNKNOWN")

        text = f"{label} {conf:.2f}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
        cv2.putText(frame, text, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Sebateurs", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
