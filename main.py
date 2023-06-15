import cv2
import os
import csv
import time

import numpy as np
from nets import nn

import torch
import warnings
from ultralytics import YOLO
from utils.util import draw_line, writes_area_text, which_area, draw_everything

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


model = YOLO('yolov8m.pt')
deepsort = nn.DeepSort()

video_path = "data/test.mp4"


def main():
	prev_frame_time = 0
	next_frame_time = 0
	areas_names = ['Register','Entrance','A1','A2','A3']
	time_spent = {}

	f = open('values.csv','w',newline='')
	writer = csv.DictWriter(f,fieldnames=areas_names)
	writer.writeheader()

	cap = cv2.VideoCapture(video_path)

	while cap.isOpened():
		success, frame = cap.read()

		if not success:
			break

		results = model.predict(frame,stream=False,show_labels=True,show_conf=False,
								save=False,imgsz=640,conf=0.35,classes=0)

		boxes = results[0].boxes
		areas = {'Register':0,'Entrance':0,'A1':0,'A2':0,'A3':0}
		boxes_d = boxes.xywh
		confs_d = boxes.conf
		boxes_d = np.array(boxes_d.cpu())
		confs_d = np.array(confs_d.cpu())
		indic_d = np.zeros_like(confs_d)

		outputs = deepsort.update(boxes_d,confs_d,indic_d,frame)
		inds = []

		if len(outputs) > 0:
			boxes = outputs[:, :4]
			object_id = outputs[:, -1]
			indentities = outputs[:, -2]
			for i, box in enumerate(boxes):
				if object_id[i] != 0:
					continue
				x1,y1,x2,y2 = list(map(int, box))
				x, y = x2-(x2-x1)/2, y2-(y2-y1)/2
				index = int(indentities[i]) if indentities is not None else 0
				inds.append(index)
				cv2.circle(frame,(int(x),int(y)),3,(255,0,0),3)
				area = which_area(frame,x,y)

				try:
					areas[area] += 1
				except KeyError:
					print("No such area")
				cv2.putText(frame,area,(int(x),int(y-10)),cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),1,cv2.LINE_AA)

				text = f'ID:{str(index)}'
				cv2.putText(frame,text,(int(x),int(y+10)),0,1,(0,255,255),2,lineType=cv2.FILLED)
		
		draw_everything(frame)
		writer.writerow(areas)

		next_frame_time = time.time()
		try:
			fps = str(round(1/(next_frame_time-prev_frame_time),2))
		except ZeroDivisionError:
			fps = ""
		prev_frame_time = next_frame_time

		cv2.putText(frame,"FPS: "+fps,(7,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3,cv2.LINE_AA)
		
		cv2.imshow("Tracking", frame)

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break


	f.close()
	cap.release()
	cv2.destroyAllWindows()



if __name__=="__main__":
	main()
