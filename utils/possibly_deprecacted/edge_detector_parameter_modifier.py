# Created by: Hunter Young
# Date: 10/5/17
#
# Script Description:
# 	Script is designed to take various commandline arguments making a very simple
# 	and user-friendly method to take any video file of interest as input and extract
# 	all the possible images available into a seperate folder, in addition to outputting
# 	a .csv file logging any additional useful information that is associated with each image.
#
# Current Recommended Usage: (in terminal)
# 	python parse_video.py -p /path/to/videos/home/directory -v name_of_video.mp4 -o name_of_output_data_file
#
# ASSUMPTIONS:
# 	Assumes that the video file name is formatted like so, "X_X_MMDDYY_TimeOfDay_SysTimeMsecRecordingBegins.mp4"

import cv2
import os
import csv
import argparse
from matplotlib import pyplot as plt
import numpy as np

def onChange(pos):
    global img
    global gray
    global tmp
    tmp = np.copy(img)

    # get current positions of four trackbars
    ks = cv2.getTrackbarPos('kernelSize','image')
    e1 = cv2.getTrackbarPos('edgeIn1','image')
    e2 = cv2.getTrackbarPos('edgeIn2','image')

    edges = cv2.Canny(tmp,e1,e2,apertureSize = 3)
    cv2.imshow("edges", edges)

    res2 = cv2.bitwise_and(img, img, mask = thresh)
    kernel = np.ones((ks,ks),np.uint8)
    opening = cv2.morphologyEx(res2,cv2.MORPH_OPEN,kernel, iterations = 2)
    cv2.imshow('opened',opening)
    closing = cv2.morphologyEx(res2,cv2.MORPH_CLOSE,kernel, iterations = 2)
    cv2.imshow('closed',closing)

    hlines = opening
    lines = cv2.HoughLinesP(hlines,1,np.pi/180,desiredThresh,minLineLength,maxLineGap)
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(tmp,(x1,y1),(x2,y2),(0,255,0),2)

#Run Main
if __name__ == "__main__" :

    # Setup commandline argument(s) structures
    ap = argparse.ArgumentParser(description='Image Segmentation')
    ap.add_argument("--pic", "-p", type=str, default='testframe.jpg', metavar='FILE', help="Name of video file to parse")
    # Store parsed arguments into array of variables
    args = vars(ap.parse_args())

    # Extract stored arguments array into individual variables for later usage in script
    _img = args["pic"]

    # create trackbars for color change
    cv2.namedWindow('image')
    cv2.namedWindow('opened')
    cv2.namedWindow('closed')
    cv2.namedWindow('edges')
    cv2.createTrackbar('kernelSize','image',0,255,onChange)
    cv2.createTrackbar('edgeIn1','image',0,1000,onChange)
    cv2.createTrackbar('edgeIn2','image',0,1000,onChange)

    img = cv2.imread(_img)	# Input video file as OpenCV VideoCapture device
    tmp = np.copy(img)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 63, 0])
    upper_red = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img, img, mask = mask)

    test = cv2.pyrMeanShiftFiltering(res,10, 45, 3)
    gray = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    while True:
        # cv2.imshow("image", tmp)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
