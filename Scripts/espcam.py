import math
import urllib.request
import cv2
import cvzone
import numpy as np
from ultralytics import YOLO


url = 'http://192.168.1.22/cam-hi.jpg'

classNames = ['10 clubs', '10 diamonds', '10 hearts', '10 spades',
              '2 clubs', '2 diamonds', '2 hearts', '2 spades',
              '3 clubs', '3 diamonds', '3 hearts', '3 spades',
              '4 clubs', '4 diamonds', '4 hearts', '4 spades',
              '5 clubs', '5 diamonds', '5 hearts', '5 spades',
              '6 clubs', '6 diamonds', '6 hearts', '6 spades',
              '7 clubs', '7 diamonds', '7 hearts', '7 spades',
              '8 clubs', '8 diamonds', '8 hearts', '8 spades',
              '9 clubs', '9 diamonds', '9 hearts', '9 spades',
              'ace clubs', 'ace diamonds', 'ace hearts', 'ace spades',
              'jack  clubs', 'jack diamonds', 'jack hearts', 'jack spades',
              'king clubs', 'king diamonds', 'king hearts', 'king spades',
              'queen clubs', 'queen diamonds', 'queen hearts', 'queen spades']

# Create a VideoCapture object
cap = cv2.VideoCapture(url)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../Yolo-Weights/runs/detect/train/weights/best.pt")

# Check if the IP camera stream is opened successfully
if not cap.isOpened():
    print("Failed to open the IP camera stream")
    exit()

# Read and display video frames
while True:
    # Read a frame from the video stream
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    # ret, frame = cap.read()
    img = cv2.imdecode(imgnp, -1)

    # cv2.imshow('live Cam Testing', im)
    results = model(img, stream=True, device="0")
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            cvzone.cornerRect(img, bbox=(x, y, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f"{classNames[cls]} {conf}",
                               (max(0, int(x1)), max(35, int(y1))),  # bboxes coordinates
                               scale=3,  # make things smaller
                               thickness=2,  # letter thickness
                               offset=5
                               )
    cv2.imshow("Image", img)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
