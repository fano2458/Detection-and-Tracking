from ultralytics import YOLO
import cv2
import time
import numpy as np


def predict_frame(model, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    print(results)


def inference(model):
    prev_frame_time = 0
    next_frame_time = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #predict_frame(model, frame)
        source = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model.predict(source, save=True, imgsz=320, conf=0.5)

        break
  

def main():
    model = YOLO("yolov8n.pt")
    inference(model)
    

if __name__ == '__main__':
    main()
