import cv2
import os
import os.path
import csv
import argparse
from matplotlib import pyplot as plt
import numpy as np
import math
import utils

def onChange(pos):
    global img
    global tmp
    tmp = np.copy(img)

    # get current positions of four trackbars
    sxmin = cv2.getTrackbarPos('threshMin','image')
    sxmax = cv2.getTrackbarPos('threshMax','image')
    thresh_flag = cv2.getTrackbarPos('threshInv','image')
    # apertureSize = cv2.getTrackbarPos('ApertureSize','image')
    # minLineLength = cv2.getTrackbarPos('LineLength','image')
    # maxLineGap = cv2.getTrackbarPos('LineGap','image')
    # desiredThresh = cv2.getTrackbarPos('Threshold','image')

    x_thresh = abs_sobel_thresh(img, orient='x', thresh_min=sxmin, thresh_max=sxmax)
    # mag_thresh = mag_thresh(img, sobel_kernel=3, mag_thresh=(70, 255))
    # dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    # s_thresh = hls_select(img,channel='s',thresh=(160, 255))
    # s_thresh_2 = hls_select(img,channel='s',thresh=(200, 240))
    #
    # white_mask = select_white(img)
    # yellow_mask = select_yellow(img)
    #
    #
    # x_thresh = abs_sobel_thresh(img, orient='x', thresh_min=10 ,thresh_max=230)
    # mag_thresh = mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 150))
    # dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    # hls_thresh = hls_select(img, thresh=(180, 255))
    # lab_thresh = lab_select(img, thresh=(155, 200))
    # luv_thresh = luv_select(img, thresh=(225, 255))
    # #Thresholding combination
    # threshholded = np.zeros_like(x_thresh)
    # threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
    #
    # threshholded = np.zeros_like(x_thresh)
    # threshholded[((x_thresh == 1)) | ((mag_thresh == 1) & (dir_thresh == 1))| (white_mask>0)|(s_thresh == 1) ]=1

    cv2.imshow('image',x_thresh)
    # cv2.imshow('image',mag_thresh)
    # cv2.imshow('image',dir_thresh)

    # cv2.imshow('image',s_thresh)
    # cv2.imshow('image',s_thresh_2)
    #
    # cv2.imshow('image',white_mask)
    # cv2.imshow('image',yellow_mask)

    # cv2.imshow('image',hls_thresh)
    # cv2.imshow('image',lab_thresh)
    # cv2.imshow('image',luv_thresh)
    # cv2.imshow('image',threshholded)

    # if thresh_flag == 0:
    #     gray2 = cv2.cvtColor(opening,cv2.COLOR_BGR2GRAY)
    # if thresh_flag == 1:
    #     gray2 = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)
    #
    # ret, helper = cv2.threshold(gray2,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)



#Run Main
if __name__ == "__main__" :

    # Setup commandline argument(s) structures
    ap = argparse.ArgumentParser(description='Image Segmentation')
    ap.add_argument("--pic", "-p", type=str, default='new_test/1.jpg', metavar='FILE', help="Name of video file to parse")
    # Store parsed arguments into array of variables
    args = vars(ap.parse_args())

    # Extract stored arguments array into individual variables for later usage in script
    _img = args["pic"]

    # create trackbars for color change
    cv2.namedWindow('image')

    cv2.createTrackbar('threshInv','image',0,1,onChange)
    cv2.createTrackbar('threshMin','image',0,255,onChange)
    cv2.createTrackbar('threshMax','image',0,255,onChange)
    # cv2.createTrackbar('kernelSize','image',0,255,onChange)
    # cv2.createTrackbar('edgeIn1','image',0,1000,onChange)
    # cv2.createTrackbar('edgeIn2','image',0,1000,onChange)
    # cv2.createTrackbar('ApertureSize','image',3,7,onChange)
    # cv2.createTrackbar('LineLength','image',0,255,onChange)
    # cv2.createTrackbar('LineGap','image',0,255,onChange)
    # cv2.createTrackbar('Threshold','image',0,1000,onChange)

    img = cv2.imread(_img)	# Input video file as OpenCV VideoCapture device
    tmp = np.copy(img)
    
    while True:
        cv2.imshow("image", tmp)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
