import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        print(x, y, w, h)
        roi_grey = gray[y:y+h, x:x+w]       # For grey image
        roi_color = frame[y:y+h, x:x+w]     # For color image
        img_item = 'image.png'
        cv2.imwrite(img_item, roi_grey)

        width = x + w
        height = y + h
        cv2.rectangle(frame, (x, y), (width, height), color=(0, 255, 0), thickness=2)

    # Display resulting frame

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

