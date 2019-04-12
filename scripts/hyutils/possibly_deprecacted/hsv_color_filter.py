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

def nothing(x):
    pass

cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('Hmin','image',107,255,nothing)
cv2.createTrackbar('Smin','image',0,255,nothing)
cv2.createTrackbar('Vmin','image',0,255,nothing)
cv2.createTrackbar('Hmax','image',255,255,nothing)
cv2.createTrackbar('Smax','image',116,255,nothing)
cv2.createTrackbar('Vmax','image',255,255,nothing)

# Setup commandline argument(s) structures
ap = argparse.ArgumentParser(description='Image Segmentation')
ap.add_argument("--pic", "-p", type=str, default='new/test/frame10.jpg', metavar='FILE', help="Name of video file to parse")
# ap.add_argument("--pic", "-p", type=str, default='40.png', metavar='FILE', help="Name of video file to parse")
# Store parsed arguments into array of variables
args = vars(ap.parse_args())

# Extract stored arguments array into individual variables for later usage in script
_img = args["pic"]
# Initialize variables for data capturing
# img = cv2.imread(_img)	# Input video file as OpenCV VideoCapture device

img = cv2.imread(_img)	# Input video file as OpenCV VideoCapture device
tmp = cv2.resize(img, (640,480))
# hsv = cv2.cvtColor(tmp,cv2.COLOR_BGR2HSV)
hsv = cv2.cvtColor(tmp,cv2.COLOR_BGR2YUV)

while(1):
	# get current positions of four trackbars
	hmin = cv2.getTrackbarPos('Hmin','image')
	smin = cv2.getTrackbarPos('Smin','image')
	vmin = cv2.getTrackbarPos('Vmin','image')
	hmax = cv2.getTrackbarPos('Hmax','image')
	smax = cv2.getTrackbarPos('Smax','image')
	vmax = cv2.getTrackbarPos('Vmax','image')
	lower_red = np.array([hmin, smin, vmin])
	upper_red = np.array([hmax, smax, vmax])

	mask = cv2.inRange(hsv, lower_red, upper_red)
	res = cv2.bitwise_and(tmp, tmp, mask = mask)

	cv2.imshow('1',tmp)
	cv2.imshow('2',mask)
	cv2.imshow('image',res)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
