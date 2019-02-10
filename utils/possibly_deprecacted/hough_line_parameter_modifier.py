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
import os.path
import csv
import argparse
from matplotlib import pyplot as plt
import numpy as np
import math

def onChange(pos):
    global img
    global tmp
    # tmp = np.copy(img)
    tmp = cv2.resize(img, (640,480))

    # get current positions of four trackbars
    hmin = cv2.getTrackbarPos('Hmin','image')
    smin = cv2.getTrackbarPos('Smin','image')
    vmin = cv2.getTrackbarPos('Vmin','image')
    hmax = cv2.getTrackbarPos('Hmax','image')
    smax = cv2.getTrackbarPos('Smax','image')
    vmax = cv2.getTrackbarPos('Vmax','image')
    ks = cv2.getTrackbarPos('kernelSize','image')
    e1 = cv2.getTrackbarPos('edgeIn1','image')
    e2 = cv2.getTrackbarPos('edgeIn2','image')
    thresh_flag = cv2.getTrackbarPos('threshInv','image')
    apertureSize = cv2.getTrackbarPos('ApertureSize','image')
    minLineLength = cv2.getTrackbarPos('LineLength','image')
    maxLineGap = cv2.getTrackbarPos('LineGap','image')
    desiredThresh = cv2.getTrackbarPos('Threshold','image')

    if apertureSize % 2 == 0:
        apertureSize += 1
    if apertureSize < 3:
        apertureSize = 3

    # lower_red = np.array([0, 0, 195])
    lower_red = np.array([hmin, smin, vmin])
    upper_red = np.array([255, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    res1 = cv2.bitwise_and(img, img, mask = mask1)
    cv2.imshow('white filtered',res1)


    lower_red2 = np.array([hmax, smax, vmax])
    upper_red2 = np.array([255, 255, 255])
    # lower_red2 = np.array([hmin, smin, vmin])
    # upper_red2 = np.array([hmax, smax, vmax])

    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    res2 = cv2.bitwise_and(img, img, mask = mask2)
    cv2.imshow('color filtered',res2)

    comp_mask = cv2.add(mask1,mask2)
    # comp_mask = cv2.subtract(mask2,mask1)
    res = cv2.add(res1, res2)

    cv2.imshow('Composite mask',comp_mask)
    cv2.imshow('Composite filtered',res)



    test = cv2.pyrMeanShiftFiltering(res,10, 45, 3)
    gray = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)

    if thresh_flag == 0:
        ret, thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    if thresh_flag == 1:
        ret, thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    edges1 = cv2.Canny(tmp,e1,e2,apertureSize = 3)
    # cv2.imshow("edges", edges1)

    res3 = cv2.bitwise_and(img, img, mask = thresh)
    kernel = np.ones((ks,ks),np.uint8)

    opening = cv2.morphologyEx(res3,cv2.MORPH_OPEN,kernel, iterations = 2)
    cv2.imshow('opened',opening)
    closing = cv2.morphologyEx(res3,cv2.MORPH_CLOSE,kernel, iterations = 2)
    cv2.imshow('closed',closing)

    if thresh_flag == 0:
        edges = cv2.Canny(opening,e1,e2,apertureSize = 3)
        gray2 = cv2.cvtColor(opening,cv2.COLOR_BGR2GRAY)
    if thresh_flag == 1:
        edges = cv2.Canny(closing,e1,e2,apertureSize = 3)
        gray2 = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)


    ret, helper = cv2.threshold(gray2,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    hlines = helper
    lines = cv2.HoughLinesP(hlines,1,np.pi/180,desiredThresh,minLineLength,maxLineGap)
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(tmp,(x1,y1),(x2,y2),(0,255,0),2)


#Run Main
if __name__ == "__main__" :

    # Setup commandline argument(s) structures
    ap = argparse.ArgumentParser(description='Image Segmentation')
    ap.add_argument("--pic", "-p", type=str, default='40.png', metavar='FILE', help="Name of video file to parse")
    # Store parsed arguments into array of variables
    args = vars(ap.parse_args())

    # Extract stored arguments array into individual variables for later usage in script
    _img = args["pic"]

    # create trackbars for color change
    cv2.namedWindow('image')
    cv2.namedWindow('opened')
    cv2.namedWindow('closed')
    # cv2.namedWindow('edges')
    cv2.createTrackbar('Hmin','image',35,255,onChange)
    cv2.createTrackbar('Smin','image',52,255,onChange)
    cv2.createTrackbar('Vmin','image',118,255,onChange)
    cv2.createTrackbar('Hmax','image',255,255,onChange)
    cv2.createTrackbar('Smax','image',255,255,onChange)
    cv2.createTrackbar('Vmax','image',255,255,onChange)
    cv2.createTrackbar('threshInv','image',0,1,onChange)
    cv2.createTrackbar('kernelSize','image',0,255,onChange)
    cv2.createTrackbar('edgeIn1','image',0,1000,onChange)
    cv2.createTrackbar('edgeIn2','image',0,1000,onChange)
    cv2.createTrackbar('ApertureSize','image',3,7,onChange)
    cv2.createTrackbar('LineLength','image',0,255,onChange)
    cv2.createTrackbar('LineGap','image',0,255,onChange)
    cv2.createTrackbar('Threshold','image',0,1000,onChange)

    img = cv2.imread(_img)	# Input video file as OpenCV VideoCapture device
    # tmp = np.copy(img)
    tmp = cv2.resize(img, (640,480))
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    while True:
        cv2.imshow("image", tmp)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
