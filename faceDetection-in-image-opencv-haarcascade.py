import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


# Capture frame-by-frame
img = cv2.imread('images/giangho.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(30, 30)
)

# Draw a rectangle around the faces
expan = 0
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x-expan, y-expan), (x+w+expan, y+h+expan), (0, 255, 0), 2)



while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Display the resulting frame
    cv2.imshow('Image', img)


# When everything is done, release the capture
cv2.destroyAllWindows()