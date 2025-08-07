import cv2 as cv
from ultralytics import YOLO
import numpy as np
import random

FRAME_BUFFER = 10
CONFIDENCE_LVL = 0.45

CURRENT_FRAMES = 0


CURRENT_RECTANGLES = []

videoCapture = cv.VideoCapture(0)

yoloModel = YOLO("yolo11l.pt")

color_cache = {}

def getItemsInFrame(frame):
    results = yoloModel.track(frame, stream=True)
    
    rectangles = []
    
    for result in results:
        for box in result.boxes:
            confidenceLevel = box.conf.cpu().numpy()[0]
            
            if confidenceLevel < CONFIDENCE_LVL:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            name = yoloModel.names.get(box.cls.item())
            color = color_cache.setdefault(name, (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            rectangles.append([x1,y1,x2,y2, f"{name} with {round(confidenceLevel, 2)*100}% Confidence", color])
        
    return rectangles

while True:
    suc, frame = videoCapture.read()
    CURRENT_FRAMES+=1
    
    if CURRENT_FRAMES%FRAME_BUFFER==0:
        CURRENT_RECTANGLES = getItemsInFrame(frame)
    
    for rectangle in CURRENT_RECTANGLES:
        x1, y1, x2, y2, name, color = rectangle
        
        cv.rectangle(frame, (x1,y1), (x2,y2),color, 3)
        cv.putText(frame, name, (x1+30,y1+30), cv.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
    
    cv.imshow("Webcam", frame)
    if not suc or (cv.waitKey(3) == ord('q')):
        break
    
videoCapture.release()
cv.destroyAllWindows()