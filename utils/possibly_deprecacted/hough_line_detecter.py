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

image_data_path = os.path.abspath('../CRBD/Images')
gt_data_path = os.path.abspath('../CRBD/GT data')
image_out_path = os.path.abspath('../output')


NUMBER_OF_STRIPS = 10             # How many strips the image is split into
SUM_THRESH = 2                    # How much green in a strip before it's a plant
DIFF_NOISE_THRESH = 8             # How close can two sections be?

HOUGH_RHO = 5                     # Distance resolution of the accumulator in pixels
HOUGH_ANGLE = math.pi/180         # Angle resolution of the accumulator in radians
HOUGH_THRESH = 6                  # Accumulator threshold parameter. Only those lines are returned that get enough votes

ANGLE_THRESH = math.pi*(30.0/180) # How steep angles the crop rows can be in radians

use_camera = False
#view_all_steps = False
images_to_save = [2, 3, 4, 5]
strip_to_save = 3
curr_image = 0
timing = False

def strip_process(image_edit):

    height = len(image_edit)
    width = len(image_edit[0])

    strip_height = height / NUMBER_OF_STRIPS
    crop_points = np.zeros((height, width), dtype=np.uint8)

    for strip_number in range(NUMBER_OF_STRIPS):
        image_strip = image_edit[(strip_number*strip_height):((strip_number+1)*strip_height-1), :]


        if strip_number == strip_to_save:
            save_image('4_image_strip_4', image_strip)

        v_sum = [0] * width
        v_thresh = [0] * width
        v_diff = [0] * width
        v_mid = [0] * width

        diff_start = 0
        diff_end = 0
        diff_end_found = True

        for col_number in range(width):

            ### Vertical Sum ###
            v_sum[col_number] = sum(image_strip[:, col_number]) / 255

            ### Threshold ###
            if v_sum[col_number] >= SUM_THRESH:
                v_thresh[col_number] = 1
            else:
                v_thresh[col_number] = 0

            ### Differential with Noise Reduction ###
            if v_thresh[col_number] > v_thresh[col_number - 1]:
                v_diff[col_number] = 1
                if (col_number - diff_end) > DIFF_NOISE_THRESH:
                    diff_start = col_number
                    diff_end_found = False

            elif v_thresh[col_number] < v_thresh[col_number - 1]:
                v_diff[col_number] = -1

                if (col_number - diff_start) > DIFF_NOISE_THRESH:
                    v_mid[diff_start + (col_number-diff_start)/2] = 1
                    diff_end = col_number
                    diff_end_found = True

        if curr_image in images_to_save and strip_number == strip_to_save:
            print(v_sum)
            print(v_thresh)
            print(v_diff)
            print(v_mid)

        crop_points[(strip_number*strip_height), :] = v_mid
        crop_points *= 255

        #image_edit[(strip_number*strip_height):((strip_number+1)*strip_height-1), :] = image_strip

    return crop_points

def save_image(image_name, image_data):
    if curr_image in images_to_save:
        image_name_new = os.path.join(image_out_path, "{0}_{1}.jpg".format(image_name, str(curr_image) ))

        cv2.imwrite(image_name_new, image_data)

def crop_point_hough(crop_points):

    height = len(crop_points)
    width = len(crop_points[0])

    #crop_line_data = cv2.HoughLinesP(crop_points, 1, math.pi/180, 2, 10, 10)
    crop_line_data = cv2.HoughLines(crop_points, HOUGH_RHO, HOUGH_ANGLE, HOUGH_THRESH)

    crop_lines = np.zeros((height, width, 3), dtype=np.uint8)

    if crop_line_data != None:
        crop_line_data = crop_line_data[0]
        #print(crop_line_data)

        if len(crop_line_data[0]) == 2:
            for [rho, theta] in crop_line_data:
                #print(rho, theta)
                if (theta <= ANGLE_THRESH) or (theta >= math.pi-ANGLE_THRESH):
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    point1 = (int(round(x0+1000*(-b))), int(round(y0+1000*(a))))
                    point2 = (int(round(x0-1000*(-b))), int(round(y0-1000*(a))))
                    cv2.line(crop_lines, point1, point2, (0, 0, 255), 2)

        elif len(crop_line_data[0]) == 4:
            for [x0, y0, x1, y1] in crop_line_data:
                cv2.line(crop_lines, (x0, y0), (x1, y1), (0, 0, 255), 2)
    else:
        print("No lines found")

    return crop_lines


def crop_row_detect(image_in):

    save_image('0_image_in', image_in)

    ### Grayscale Transform ###
    image_edit = grayscale_transform(image_in)
    save_image('1_image_gray', image_edit)

    ### Binarization ###
    _, image_edit = cv2.threshold(image_edit, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    save_image('2_image_bin', image_edit)

    ### Stripping ###
    crop_points = strip_process(image_edit)
    save_image('8_crop_points', crop_points)

    ### Hough Transform ###
    crop_lines = crop_point_hough(crop_points)
    save_image('9_image_hough', cv2.addWeighted(image_in, 1, crop_lines, 1, 0.0))

    return crop_lines


def grayscale_transform(image_in):
    b, g, r = cv2.split(image_in)
    return 2*g - r - b

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
    img = cv2.imread(_img)	# Input video file as OpenCV VideoCapture device
    tmp = np.copy(img)

    gray = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(tmp,cv2.COLOR_BGR2HSV)
    yuv = cv2.cvtColor(tmp,cv2.COLOR_BGR2YUV)

    # lower_hsv = np.array([35, 52, 118])
    # upper_hsv = np.array([255, 255, 255])

    # lower_yuv = np.array([107, 0, 0])
    # upper_yuv = np.array([255, 116, 255])

    lower_yuv = np.array([0, 0, 0])
    upper_yuv = np.array([164, 126, 255])

    mask = cv2.inRange(yuv, lower_yuv, upper_yuv)
    res = cv2.bitwise_and(tmp, tmp, mask = mask)
    plt.imshow(res, cmap='gray')

    gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    plt.imshow(thresh, cmap='gray')
    plt.show()

    crop_lines = crop_row_detect(thresh)

    while True:
        cv2.imshow("Alternative",crop_lines)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
