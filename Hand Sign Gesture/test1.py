import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from math import ceil
import time
from cvzone.ClassificationModule import Classifier
import tensorflow

offset = 20
imgSize = 300
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
folder = "Images/C"
counter = 0
labels = ['A', "B", "C", "1"]
classifier = Classifier('Models/keras_model.h5', 'Models/labels.txt')

while True:
    ret, frame = cap.read()
    imgOutput = frame.copy()
    hands, frame = detector.findHands(frame)

    #Crop The Image
    if hands:
        hand = hands[0]
        # print("Hand: ", hand['bbox'])
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        x, y, w, h = hand['bbox']
        imgCrop = frame[y-offset:y+h+offset, x-offset:x+w+offset]

        # img White 1st parameter = height, 2nd parameter = width
        # imgWhite[0:imgCrop.shape[0], 0:imgCrop.shape[1]] = imgCrop

        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h
            calW = ceil(k*w)

            imgResize = cv2.resize(imgCrop, (calW, imgSize))
            wGap = ceil((imgSize-calW)/2)
            imgWhite[:, wGap:calW+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            # print(prediction, index)

        else:
            k = imgSize/w
            calH = ceil(k*h)

            imgResize = cv2.resize(imgCrop, (imgSize, calH))
            wGap = ceil((imgSize-calH)/2)
            imgWhite[wGap:calH+wGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        cv2.rectangle(imgOutput, (x-offset, y-offset-50), ((x-offset)+100, y-offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255,250,255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)

        cv2.imshow("White Image", imgWhite)

        # cv2.imshow("Hand Image", imgCrop)
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)