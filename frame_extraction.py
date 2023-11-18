import cv2
import os

vid = cv2.VideoCapture('/Users/sanshubhkukutla/Documents/projects/RoadBrain/traffic.mov')
currentframe = 0

if not os.path.exists('data'):
    os.makedirs('data')

while True:
    success, frame = vid.read()

    cv2.imshow('Output', frame)
    cv2.imwrite('./data/frame' + str(currentframe) + '.jpeg', frame)
    currentframe +=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()