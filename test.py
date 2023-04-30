import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

video_path = "data/test.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    results = model.predict(frame, save=False, imgsz=640, conf=0.35, classes=0)
    annotated_frame = results[0].plot()
    
    boxes = results[0].boxes
    points = []
    for box in boxes:
        x1,y1,x2,y2 = box.xyxy[0].detach().cpu()
        x, y = x2-(x2-x1)/2, y2-(y2-y1)/2
        cv2.circle(annotated_frame,(int(x),int(y)),3,(255,0,0),3)

    cv2.imshow("YOLOv8 Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
