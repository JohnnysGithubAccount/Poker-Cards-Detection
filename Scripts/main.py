from ultralytics import YOLO
import cvzone
import cv2
import math


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../Yolo-Weights/runs/detect/train/weights/best.pt")

classNames = ['10 of clubs', '10 of diamonds', '10 of hearts', '10 of spades',
              '2 of clubs', '2 of diamonds', '2 of hearts', '2 of spades',
              '3 of clubs', '3 of diamonds', '3 of hearts', '3 of spades',
              '4 of clubs', '4 of diamonds', '4 of hearts', '4 of spades',
              '5 of clubs', '5 of diamonds', '5 of hearts', '5 of spades',
              '6 of clubs', '6 of diamonds', '6 of hearts', '6 of spades',
              '7 of clubs', '7 of diamonds', '7 of hearts', '7 of spades',
              '8 of clubs', '8 of diamonds', '8 of hearts', '8 of spades',
              '9 of clubs', '9 of diamonds', '9 of hearts', '9 of spades',
              'ace of clubs', 'ace of diamonds', 'ace of hearts', 'ace of spades',
              'jack  of clubs', 'jack of diamonds', 'jack of hearts', 'jack of spades',
              'king of clubs', 'king of diamonds', 'king of hearts', 'king of spades',
              'queen of clubs', 'queen of diamonds', 'queen of hearts', 'queen of spades']

while True:
    success, img = cap.read()
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
                               )
    cv2.imshow("Image", img)
    cv2.waitKey(1)
