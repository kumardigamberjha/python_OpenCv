'''###################################################################################
                            Open CV Hand Gesture
###################################################################################'''

####################### Importing Modules ################################
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from math import ceil
import time

offset = 20
imgSize = 300
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
folder = "Images/1"
counter = 0

while True:
    ret, frame = cap.read()
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

        else:
            k = imgSize/w
            calH = ceil(k*h)

            imgResize = cv2.resize(imgCrop, (imgSize, calH))
            wGap = ceil((imgSize-calH)/2)
            imgWhite[wGap:calH+wGap, :] = imgResize

        cv2.imshow("White Image", imgWhite)

        # cv2.imshow("Hand Image", imgCrop)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)