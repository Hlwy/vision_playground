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
