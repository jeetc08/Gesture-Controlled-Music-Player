import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp 
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time 

base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
options = mp_vision.HandLandmarkerOptions(
base_options=base_options,
running_mode=mp_vision.RunningMode.VIDEO,
num_hands=2,
)

landmarker = mp_vision.HandLandmarker.create_from_options(options)

webcam = cv2.VideoCapture(0)

starttime = time.time()

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky finger
    (0, 17)                                 # Palm base
]
def hand(frame,hand_landmarks):
    h,w,_ = frame.shape
    points = [(int(lm.x*w),int(lm.y * h)) for lm in hand_landmarks]
    for i,j in HAND_CONNECTIONS:
        cv2.line(frame, points[i], points[j], (0,255,0), 2)
    for x,y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


while True:
    ret,frame = webcam.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = rgb)

    timestamp = int((time.time()-starttime)*1000)

    result = landmarker.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:
        print(f"Hands detetected: {len(result.hand_landmarks)})")
    else:
        print("Hands detected: 0")

    count = 0

    if result.hand_landmarks:
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            hand(frame, hand_landmarks)

            hand_label = result.handedness[i][0].category_name

            if hand_label == "Right":
                if hand_landmarks[4].x > hand_landmarks[3].x: count += 1
            else:
                if hand_landmarks[4].x < hand_landmarks[3].x: count += 1


            if hand_landmarks[8].y < hand_landmarks[6].y: count += 1
            if hand_landmarks[12].y < hand_landmarks[10].y: count += 1
            if hand_landmarks[16].y < hand_landmarks[14].y: count += 1
            if hand_landmarks[20].y < hand_landmarks[18].y: count += 1
    cv2.putText(frame, str(count), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break


webcam.release()
cv2.destroyAllWindows()
landmarker.close()  