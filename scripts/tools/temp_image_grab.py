import cv2, time
import numpy as np
import matplotlib.pyplot as plt

flag_save = 0
cap = cv2.VideoCapture("realsense_rgb_2.avi")

count = 0
keyFrame = 10
dImg = None
try:
    while True:
        if count < keyFrame:
            # Show images
            _,dImg = cap.read()
            count+=1
        elif count == keyFrame:
            cv2.imwrite("test_rgb.png",dImg)
            cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
            cv2.imshow('Disparity', dImg)
            cv2.waitKey(1)
        else:
            continue


finally:
    if flag_save:
        writerRGB.release()
        writerD.release()
cv2.waitKey(1)
