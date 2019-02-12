# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO

import numpy as np
import os, sys
import cv2

def line_finder(
    _img="test/test_disparity.png", method = 1, line_input_method = 0, e1 = 2, e2 = 5,
    ang = 90, rho = 1, minLineLength = 11, maxLineGap = 11, houghThresh = 50, greyThresh = 11,
    cnt_thresh = 500.0, show_helpers = True, use_umap = True, flip_thresh_bin = False
    ):

    global filtered, pre_filter, umap, vmap, cpy1, cpy2, overlay, helper_img, prev_img
    global flag_new_img, last_img, prev_e1, prev_e2, prev_greyThresh, prev_map

    """
    Parse Input Variables to Create string desscriptors for easier control variable copying
    """
    if(line_input_method is 1): filter_meth = "Composite Filtering -> Blurring"
    elif(line_input_method is 2): filter_meth = "Composite Filtering -> Canny Edges"
    else: filter_meth = "Basic Thresholding"

    if(method is 0): line_meth = "Standard Hough Transform"
    else: line_meth = "Probablistic Hough Transform"

    if(use_umap): map_space = "U-Map"
    else: map_space = "V-Map"

    """
    Check control variable arguments for changes to reduce repeating unnecessary calculations
    """

    # Check if we are testing for a different image
    if(last_img == _img): flag_new_img = False
    else: flag_new_img = True

    # Check if we need to perform image filtering again
    if((prev_map is not use_umap) or (flag_new_img) or
       (prev_greyThresh is not greyThresh) or
       (prev_e1 is not e1) or
       (prev_e2 is not e2)
    ):
        flag_needs_filtering = True
        print("Images need to be filtered again...")
    else:
        flag_needs_filtering = True
        print("Skipping image filtering...")

    last_img, prev_e1, prev_e2, prev_greyThresh, prev_map = _img, e1, e2, greyThresh, use_umap

    print(
    """
    Inputs:  (image = %s)
    ------

    \t* Mapping Space       : %s
    \t* Line Finding Method : %s
    \t* Filtering Used      : %s
    \t* Kernel Size         : (%d, %d)
    \t* Rho, Angle (deg)    :  %d, %d
    \t* Min Line Length     :  %d
    \t* Max Line Gap        :  %d
    \t* Grey Thresholding   :  %d
    \t* Hough Threshold     :  %d
    \t* Contour Threshold   :  %d
    """ % (
        _img,map_space,line_meth,filter_meth,e1,e2,rho,ang,minLineLength,maxLineGap,greyThresh,houghThresh,cnt_thresh
    ))

    # Convert Angle to radians for calculations
    ang = ang * np.pi/180
    img = cv2.imread(_img,cv2.IMREAD_GRAYSCALE)
    if(flag_new_img):
        print("Mapping new image into UV Map")
        umap,vmap, overlay = uvMapping(img)

    if(use_umap): tmpIn = np.copy(umap)
    else: tmpIn = np.copy(vmap)

    tmp = cv2.cvtColor(tmpIn,cv2.COLOR_GRAY2BGR)
    n,m,_ = tmp.shape
    filtered = np.zeros((n,m,3),dtype=np.uint8)

    if(flag_needs_filtering):
        if(flip_thresh_bin): masking = cv2.THRESH_BINARY_INV
        else: masking = cv2.THRESH_BINARY

        kernel = np.ones((e1,e2),np.uint8)
        kernel2 = np.ones((4,4),np.uint8)

        ret, grey = cv2.threshold(tmpIn,greyThresh,255,masking)
        dilation = cv2.dilate(grey,kernel,iterations = 1)
        blur = cv2.GaussianBlur(dilation,(5,5),0)
        closing = cv2.morphologyEx(grey,cv2.MORPH_CLOSE,kernel, iterations = 2)

        ret, grey_thresh = cv2.threshold(grey,greyThresh,255,masking)
        ret, close_thresh = cv2.threshold(closing,greyThresh,255,masking)
        canny = cv2.Canny(blur,25,200,apertureSize = 3)

        helper_imgs = [tmp,grey,dilation,blur,closing,canny]
        helper_img = construct_helper_img(helper_imgs)

    if(line_input_method is 1): hlines = blur
    elif(line_input_method is 2): hlines = canny
    else: hlines = grey
#     cv2.imwrite('input-lines.png',hlines)
    try:
        if(method==0):
            lines = cv2.HoughLines(hlines,rho,ang,houghThresh)
            count = 0
            for rho,theta in lines[0]:
                count+=1
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(tmp,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.line(filtered,(x1,y1),(x2,y2),(255,255,255),2)
        else:
            lines = cv2.HoughLinesP(hlines,rho,ang,houghThresh,minLineLength,maxLineGap)
            for x in range(0, len(lines)):
                for x1,y1,x2,y2 in lines[x]:
                    cv2.line(tmp,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.line(filtered,(x1,y1),(x2,y2),(255,255,255),2)
    except:
        print("Couldn't Find Hough Lines!!!")
        pass

    if(flag_new_img): plot_image(overlay,0)
    if(flag_needs_filtering and show_helpers): plot_image(helper_img,1)
    try: plot_image(filtered,2)
    except: pass

    # CONTOUR SECTIION
    pre_filter = cv2.GaussianBlur(filtered,(5,5),0)
#     try: cv2.imwrite('pre-filtered-lines.png',pre_filter)
#     except: pass
    pre_filter = cv2.cvtColor(pre_filter,cv2.COLOR_BGR2GRAY)
    abstract_horizontals(pre_filter,cnt_thresh)

    tmpI = np.copy(img)
    indices = np.argwhere(pre_filter == 255)
#     print(indices.shape)
    indices = indices[:,:2]
    disparity_filters = indices[:,0]
#     pp.pprint(disparity_filters)

    tmpI[np.isin(tmpI, disparity_filters)] = 0
    plot_image(pre_filter,8)
    plot_image(cv2.cvtColor(tmpI,cv2.COLOR_GRAY2BGR),9)

    print(" ============  Plotting Done !! =================")
