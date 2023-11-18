import cv2
import os

webCam = cv2.VideoCapture(0)
currentframe = 0

while True:
    success, frame = webCam.read()

    cv2.imshow('Output', frame)
    cv2.imwrite('frame' + str(currentframe) + '.jpeg', frame)
    currentframe +=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

