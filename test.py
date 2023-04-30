import cv2
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

video_path = "data/test.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.predict(frame, save=False, imgsz=640, conf=0.35, classes=0)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
