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


def abstract_horizontals(_img,threshold, show_contours=True,verbose=False):
    try: img = cv2.cvtColor(_img,cv2.COLOR_BGR2GRAY)
    except: img = _img
    n,m = img.shape[0], img.shape[1]
    print("Countours:", img.shape)
    _, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]

    filtered_cnts = [cnt for cnt in contours if cv2.contourArea(cnt) >= threshold]
    filtered_areas = [cv2.contourArea(cnt) for cnt in filtered_cnts]

    # Find horizontal limits
    limits = []; ellipses = []; pts = []; centers = []
    count = 0
    for cnt in filtered_cnts:
        ellipse = cv2.fitEllipse(cnt)
        cx, cy = np.int32(ellipse[0])
        h, w = np.int32(ellipse[1])
        dx,dy = w/2,h/2

        if(verbose):
            print("Center: (%d, %d)" % (cx,cy))
            print("H, W: (%d, %d)" % (dx,dy))

        xr = cx + dx; xl = cx - dx
        yt = cy - dy; yb = cy + dy
        ptL = (xl,cy); ptR = (xr,cy)

        lim = np.array([[xl,xr],[yt,yb]])
        limits.append(lim)

        pts.append([ptL, ptR])
        ellipses.append(ellipse)
        centers.append((cx,cy))
        count+=1

    if(verbose):
        print("Raw Contour Areas:",areas)
        print("Filtered Contour Areas:",filtered_areas)

    
    if(show_contours):
        helper1 = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        helper2 = np.copy(helper1)
        helper_in = np.copy(helper1)
        for i in range(count):
            cnt = filtered_cnts[i]
            cv2.drawContours(helper1, [cnt], 0, (255,255,255), 2)
            cv2.ellipse(helper2,ellipses[i],(0,255,0),2)
            cv2.circle(helper2,centers[i],2,(255,0,255), 5)
            cv2.circle(helper2,pts[i][0],2,(0,255,255), 5)
            cv2.circle(helper2,pts[i][1],2,(0,255,255), 5)

        sborder = np.ones((n,5,3),dtype=np.uint8)
        sborder[np.where((sborder==[1,1,1]).all(axis=2))] = [255,0,255]
        helper = np.concatenate((
            np.concatenate((helper_in,sborder), axis=1),
            np.concatenate((helper1,sborder), axis=1),
            np.concatenate((helper2,sborder), axis=1)
        ), axis=1)
        plot_image(helper,3)

    return filtered_cnts, limits, ellipses, pts, centers
