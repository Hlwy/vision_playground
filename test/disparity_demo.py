#!/usr/bin/env python
import cv2, time
import numpy as np
import argparse, pprint

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import Bbox
import matplotlib.gridspec as gridspec

pp = pprint.PrettyPrinter(indent=4)

def uvMapping(_img, get_overlay=True,verbose=False):
    overlay = None
    h,w = _img.shape
    histRange = (0,256)
    histSz = np.max(_img) + 1
    if(verbose): print("[UV Mapping] Input Image Size: (%d, %d)" % (h,w))

    umap = np.zeros((histSz,w,1), dtype=np.uint8)
    vmap = np.zeros((h,histSz,1), dtype=np.uint8)

    for i in range(0,h):
        vscan = _img[i,:]
        vrow = cv2.calcHist([vscan],[0],None,[histSz],histRange)
        if(verbose): print("\t[V Mapping] Scan [%d] (%s) ---- Scan Histogram (%s)" % (i,', '.join(map(str, vscan.shape)), ', '.join(map(str, vrow.shape))))
        vmap[i,:] = vrow

    for i in range(0,w):
        uscan = _img[:,i]
        urow = cv2.calcHist([uscan],[0],None,[histSz],histRange)
        if(verbose): print("\t[U Mapping] Scan[%d] (%s) ---- Scan Histogram (%s)" % (i,', '.join(map(str, uscan.shape)), ', '.join(map(str, urow.shape))))
        umap[:,i] = urow

    umap = np.reshape(umap,(histSz,w))
    vmap = np.reshape(vmap,(h,histSz))

    if(get_overlay):
        blank = np.ones((umap.shape[0],vmap.shape[1]),np.uint8)*255
        pt1 = np.concatenate((_img, vmap), axis=1)
        pt2 = np.concatenate((umap,blank),axis=1)
        overlay = np.concatenate((pt1,pt2),axis=0)
        overlay = cv2.cvtColor(overlay,cv2.COLOR_GRAY2BGR)

    if(verbose):
        print("\t[UV Mapping] U Map = (%s) ----- V Map = (%s)" % (', '.join(map(str, umap.shape)),', '.join(map(str, vmap.shape)) ))
    return umap,vmap,overlay

def show_overlay(overlay):
    plt.imshow(overlay)
    plt.yticks([''])
    plt.xticks([''])

def method1(img,angle=120,rho=141,greyThresh=11,houghThresh=5,lineGap=4,lineLength=14,kernelSize=(5,5),show_helpers=False):
    copy = np.copy(img)
    tmp = cv2.cvtColor(copy,cv2.COLOR_GRAY2BGR)

    # get current positions of four trackbars
    n,m,_ = tmp.shape
    filtered = np.zeros((n,m,3))

    ret, grey = cv2.threshold(copy,greyThresh,255,cv2.THRESH_BINARY)

    kernel = np.ones(kernelSize,np.uint8)
    closing = cv2.morphologyEx(grey,cv2.MORPH_CLOSE,kernel, iterations = 2)

    ret, helper = cv2.threshold(grey,greyThresh,255,cv2.THRESH_BINARY)
    ret, helper3 = cv2.threshold(closing,greyThresh,255,cv2.THRESH_BINARY)
    hlines = helper3
    ang = angle * np.pi/180

    lines = cv2.HoughLinesP(hlines,rho,ang,houghThresh,lineLength,lineGap)
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(tmp,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.line(filtered,(x1,y1),(x2,y2),(255,255,255),2)

    if(show_helpers):
        cv2.imshow('threshold grey',grey)
        cv2.imshow('closed',closing)
        cv2.imshow('threshold',helper)
        cv2.imshow('threshold closed',helper3)

    cv2.imshow('Probablistic Lines',tmp)
    cv2.imshow('Probablistic Filtered',filtered)
    return filtered

def onChange(pos):
    global img
    global tmp
    show_helpers = False
    tmp2 = np.copy(img)
    tmp = cv2.cvtColor(tmp2,cv2.COLOR_GRAY2BGR)

    # tmp3 = cv2.cvtColor(tmp2,cv2.COLOR_BGR2GRAY)
    # tmp = cv2.cvtColor(tmp2,cv2.COLOR_BGR2GRAY)
    # tmp = cv2.resize(img, (640,480))

    # get current positions of four trackbars
    n,m,_ = tmp.shape
    filtered = np.zeros((n,m,3))
    e1 = cv2.getTrackbarPos('edgeIn1','image')
    e2 = cv2.getTrackbarPos('edgeIn2','image')
    ang2 = cv2.getTrackbarPos('angle','image')
    rho = cv2.getTrackbarPos('rho','image')
    minLineLength = cv2.getTrackbarPos('LineLength','image')
    maxLineGap = cv2.getTrackbarPos('LineGap','image')
    desiredThresh = cv2.getTrackbarPos('Threshold','image')
    desiredThresh2 = cv2.getTrackbarPos('Threshold2','image')

    ret, grey = cv2.threshold(tmp2,desiredThresh2,255,cv2.THRESH_BINARY)

    kernel = np.ones((e1,e2),np.uint8)
    closing = cv2.morphologyEx(grey,cv2.MORPH_CLOSE,kernel, iterations = 2)

    ret, helper = cv2.threshold(grey,desiredThresh2,255,cv2.THRESH_BINARY)
    ret, helper3 = cv2.threshold(closing,desiredThresh2,255,cv2.THRESH_BINARY)
    hlines = cv2.Canny(helper3,50,150,apertureSize = 3)
    ang = ang2 * np.pi/180
    cv2.imshow('edges',hlines)
    # print(ang2, ang)
    lines = cv2.HoughLines(hlines,rho,ang,desiredThresh)
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

    print(count)

    if(show_helpers):
        cv2.imshow('threshold grey',grey)
        cv2.imshow('closed',closing)
        cv2.imshow('threshold',helper)
        cv2.imshow('threshold closed',helper3)
    cv2.imshow('lines',tmp)
    cv2.imshow('filtered',filtered)

def crap1():
    pp = pprint.PrettyPrinter(indent=4)

    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    cv2.imshow('Disparity', img)

    cv2.line(histImage, ( bin_w*(i-1), hist_h - int(round(b_hist[i-1])) ),
                ( bin_w*(i), hist_h - int(round(b_hist[i])) ),
                ( 255, 0, 0), thickness=2)

    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

def crap2():
    new_dimg = np.zeros((h,w,1), dtype=np.uint8)
    new_dimg2 = np.zeros((h,w,1), dtype=np.uint8)

    for i in range(0,h):
        asa = cv2.calcBackProject([img[i,:]],[0],umap[:,i],histRange,1)
        new_dimg[i,:] = asa

    for i in range(0,h):
        asa = cv2.calcBackProject([img[:,i]],[0],vmap[i,:],histRange,1)
        new_dimg2[:,i] = asa

        print(new_dimg2.shape)


    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Vmap', cv2.WINDOW_NORMAL)
    cv2.imshow('Disparity', new_dimg)
    cv2.imshow('Vmap', new_dimg2)

    while True:
        if cv2.waitKey(1) == ord('q'):
            break

# cv2.imwrite("test_umap_tmp.png",umap)
# cv2.imwrite("test_vmap_tmp.png",vmap)


if __name__ == "__main__" :

    # Setup commandline argument(s) structures
    ap = argparse.ArgumentParser(description='Image Segmentation')
    ap.add_argument("--pic", "-p", type=str, default="test_disparity.png", metavar='FILE', help="Name of video file to parse")
    # Store parsed arguments into array of variables
    args = vars(ap.parse_args())

    # Extract stored arguments array into individual variables for later usage in script
    _img = args["pic"]

    # create trackbars for color change
    cv2.namedWindow('image')
    # cv2.namedWindow('edges')
    cv2.createTrackbar('edgeIn1','image',5,1000,onChange)
    cv2.createTrackbar('edgeIn2','image',5,1000,onChange)
    cv2.createTrackbar('rho','image',141,255,onChange)
    cv2.createTrackbar('angle','image',120,180,onChange)
    cv2.createTrackbar('LineLength','image',14,255,onChange)
    cv2.createTrackbar('LineGap','image',4,255,onChange)
    cv2.createTrackbar('Threshold','image',5,255,onChange)
    cv2.createTrackbar('Threshold2','image',11,255,onChange)

    img = cv2.imread(_img,cv2.IMREAD_GRAYSCALE)	# Input video file as OpenCV VideoCapture device
    # img = cv2.imread(_img)	# Input video file as OpenCV VideoCapture device
    # plt.ion()
    umap, vmap, overlay = uvMapping(img)


    # tmp = np.copy(img)
    # tmp = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    # h,w = img.shape
    # print(img.shape)
    #
    # flag_save = 0
    # count = 0
    #
    #
    while True:
        key = cv2.waitKey(5) & 0xFF
        cv2.imshow("image", overlay)
        if key == ord('q'):
            break
        # show_overlay(overlay)
        # plt.show()
        # plt.pause(0.01)

    cv2.destroyAllWindows()
