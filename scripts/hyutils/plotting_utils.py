# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO
import cv2
import os, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

def construct_helper_img(_imgs,cspace=cv2.COLOR_GRAY2BGR):
    print(_imgs[0].shape)
    n,m = _imgs[0].shape[0], _imgs[0].shape[1]
    print(n,m)

    bborder = np.ones((5,m,3),dtype=np.uint8)
    sborder = np.ones((n,5,3),dtype=np.uint8)
    cborder = np.ones((5,5,3),dtype=np.uint8)
    bborder[np.where((bborder==[1,1,1]).all(axis=2))] = [255,0,255]
    sborder[np.where((sborder==[1,1,1]).all(axis=2))] = [255,0,255]
    cborder[np.where((cborder==[1,1,1]).all(axis=2))] = [255,0,255]

    imgs = []
    for img in _imgs:
        im = None
        try: im = cv2.cvtColor(img,cspace)
        except: im = img
        imgs.append(im)

    helper_img_t = np.concatenate(
        ( np.concatenate((imgs[0],sborder), axis=1),
          np.concatenate((imgs[1],sborder), axis=1),
          np.concatenate((imgs[2],sborder), axis=1) ), axis=1
    )

    helper_img_m = np.concatenate(
        ( np.concatenate((bborder,cborder), axis=1),
          np.concatenate((bborder,cborder), axis=1),
          np.concatenate((bborder,cborder), axis=1) ), axis=1
    )

    helper_img_b = np.concatenate(
        ( np.concatenate((imgs[3],sborder), axis=1),
          np.concatenate((imgs[4],sborder), axis=1),
          np.concatenate((imgs[5],sborder), axis=1) ), axis=1
    )

    helper_img = np.concatenate((helper_img_t,helper_img_m,helper_img_b), axis=0)
    return helper_img

def plot_image(img,figNum=None):
    if(figNum == None): plt.figure()
    else: plt.figure(figNum)

    plt.imshow(img)
    plt.subplots_adjust(wspace=0.0,hspace=0.0,left=0.0,right=1.0,top=1.0, bottom=0.0)
    plt.show()

def pplot(img,num = 0):
    plt.figure(num)
    plt.imshow(img)
    plt.subplots_adjust(wspace=0.0,hspace=0.0,left=0.0,right=1.0,top=1.0, bottom=0.0)
    plt.show()

def pplots(imgs,title="Temp",size=(6,1),flag_resize=False,scale=(2,1)):
    size[0]
    plt.figure(title,figsize = size)
    gs1 = gridspec.GridSpec(size[0], size[1])
    gs1.update(wspace=0.0, hspace=0.0,left=0.0,right=1.0,top=1.0, bottom=0.0)

    for i, img in enumerate(imgs):
        ax1 = plt.subplot(gs1[i])
        plt.axis('on')
        sz = img.shape
        if flag_resize:
            sz2 = (sz[1]*scale[1], sz[0]*scale[0])
            if len(sz) == 3: sz2 = sz2 + (3,)
            rsize = cv2.resize(img,sz2)
        else: rsize = img
        ax1.imshow(rsize,interpolation='bilinear')
#         ax1.imshow(rsize,interpolation='nearest')
#         ax1.imshow(rsize,interpolation='bicubic')

        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')

    plt.show()
