import cv2
import numpy as np
from ultralytics import YOLO
from utils import draw_line, writes_area_text, which_area

"""
TODO 
1 split frame into sectors
2 find to which sector does the given object belong
"""

model = YOLO('yolov8m.pt')

video_path = "data/test.mp4"
cap = cv2.VideoCapture(video_path)


def draw_everything():
    draw_line(annotated_frame, 0.00, 0.20, 0.10, 0.20) # Top-left line
    draw_line(annotated_frame, 0.10, 0.25, 0.55, 0.05) # Top-middle line
    draw_line(annotated_frame, 0.10, 0.25, 0.30, 0.80) # Left line
    draw_line(annotated_frame, 0.35, 0.15, 0.65, 0.45) # Middle Line
    draw_line(annotated_frame, 0.30, 0.80, 0.85, 0.25) # Bottom line
    draw_line(annotated_frame, 0.55, 0.05, 0.85, 0.25) # Right line

    writes_area_text(annotated_frame, "Register", 0.01, 0.25)
    writes_area_text(annotated_frame, "Area 2 (A2)", 0.20, 0.05)
    writes_area_text(annotated_frame, "Area 3 (A3)", 0.30, 0.40)
    writes_area_text(annotated_frame, "Entrance", 0.70, 0.80)
    writes_area_text(annotated_frame, "Area 1 (A1)", 0.60, 0.20)


while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    results = model.predict(frame, save=False, imgsz=640, conf=0.35, classes=0)
    annotated_frame = results[0].plot()
    boxes = results[0].boxes
    
    for box in boxes:
        x1,y1,x2,y2 = box.xyxy[0].detach().cpu()
        x, y = x2-(x2-x1)/2, y2-(y2-y1)/2
        cv2.circle(annotated_frame,(int(x),int(y)),3,(255,0,0),3)

    draw_everything()

    cv2.imshow("YOLOv8 Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
