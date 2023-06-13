import cv2
import csv
import time
import numpy as np
from ultralytics import YOLO
from utils import draw_line, writes_area_text, which_area


model = YOLO('yolov8x.pt')

#video_path = r"C:\Users\fano\Downloads\Top down view of people walking.mp4"
#video_path = "data/test.mp4"
video_path = r"C:\Users\fano\Downloads\People tracking with kalman filter and yolo.mp4"
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


prev_frame_time = 0
next_frame_time = 0
areas_names = ['Register','Entrance','A1','A2','A3']

f = open('values.csv','w',newline='')
writer = csv.DictWriter(f,fieldnames=areas_names)
writer.writeheader()

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    #results = model.predict(frame, save=False, imgsz=640, conf=0.35, classes=0)
    results = model.track(frame,conf=0.35,classes=0,tracker=r"C:\Users\fano\Downloads\bytetrack.yaml")

    annotated_frame = results[0].plot()
    boxes = results[0].boxes
    areas = {'Register':0,'Entrance':0,'A1':0,'A2':0,'A3':0}

    for box in boxes:
        x1,y1,x2,y2 = box.xyxy[0].detach().cpu()
        x, y = x2-(x2-x1)/2, y2-(y2-y1)/2
        cv2.circle(annotated_frame,(int(x),int(y)),3,(255,0,0),3)
        area = which_area(annotated_frame,x,y)
        try:
            areas[area] += 1
        except KeyError:
            print("No such area")
        cv2.putText(annotated_frame, area, (int(x),int(y-10)),cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2,cv2.LINE_AA)

    #draw_everything()
    writer.writerow(areas)

    next_frame_time = time.time()
    try:
        fps = str(round(1/(next_frame_time-prev_frame_time),2))
    except ZeroDivisionError:
        fps = ""
    prev_frame_time = next_frame_time

    cv2.putText(annotated_frame,"FPS: "+fps,(7,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3,cv2.LINE_AA)
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

f.close()
cap.release()
cv2.destroyAllWindows()
