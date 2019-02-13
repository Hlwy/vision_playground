import os, sys, fnmatch
import cv2, time
import numpy as np
import argparse, pprint
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import Bbox
import matplotlib.gridspec as gridspec

flag_new_img = False
last_img = None
umap = None
vmap = None
overlay = None
helper_img = None
filtered = None
cpy1 = None
cpy2 = None
pre_filter = None

prev_e1,prev_e2,prev_greyThresh = 0, 0, 0
prev_map = False

def grab_dir_images(_dir, patterns = ['*png','*jpg'],verbose=False):
    found = []
    for root, dirs, files in os.walk(_dir):
        for pat in patterns:
            for file in files:
                if fnmatch.fnmatch(file, pat):
                    found.append(os.path.join(root, file))

    if(verbose): print(found)
    return found

def plot_image(img,figNum=None):
    if(figNum == None): plt.figure()
    else: plt.figure(figNum)

    plt.imshow(img)
    plt.subplots_adjust(wspace=0.0,hspace=0.0,left=0.0,right=1.0,top=1.0, bottom=0.0)
    plt.show()
    return 0


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

    if(True):
        print("Raw Contour Areas:",areas)
        print("Filtered Contour Areas:",filtered_areas)

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
            plot_image(helper,5)

    return filtered_cnts, limits, ellipses, pts, centers

def window_slider(_img,_filtered):
    verbose = False
    try: img = cv2.cvtColor(_filtered,cv2.COLOR_GRAY2BGR)
    except: img = _img

    try: grey = cv2.cvtColor(_filtered,cv2.COLOR_BGR2GRAY)
    except: grey = _filtered
    display = None

    print(img.shape)

    good_inds = []
    temp = np.copy(img)
    h,w = temp.shape[:2]
    black = np.zeros((h,w,3),dtype=np.uint8)

    deadzone_x = 5
    rows = [h-50, h]
    cols = [deadzone_x,w]
    axis = 0
    hist = np.sum(img[rows[0]:rows[1],cols[0]:cols[1]], axis=axis)
    plt.figure(6)
    plt.plot(range(hist.shape[0]),hist[:])
    plt.show()
    x0 = np.argmax(hist[:,0])
    location = [h,x0]

    try:
        d = display.dtype
        display_windows = np.copy(display)
    except:
        display_windows = np.copy(img)

    size = [5,15]
    mask_size = [10,15]
    window_height,window_width0 = size
    x_current = abs(int(location[1]))
    y_current = abs(int(location[0]))

    if x_current <= window_width0: x_current = window_width0
    if y_current >= h: y_current = h - window_height
    if verbose == True: print("Starting Location: ", x_current, y_current)

    nonzero = temp.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # print(len(nonzerox))
    direction = "up"

    window = 0
    flag_done = False
    threshold = 30
    x_subset = []
    y_subset = []
    subset = []


    while(window <= 40 and not flag_done):
        prev_good_inds = good_inds

        if x_current >= w/2: window_width = window_width0*2
    #     elif x_current >= w/3 and x_current < w/2: window_width = window_width0*2
        else: window_width = window_width0

        if x_current >= w:
            flag_done = True
            if verbose == True: print("Exiting: Reached max image width.")

        if y_current - window_height >= 0: win_y_low = y_current - window_height
        else: win_y_low = 0

        if y_current + window_height <= h: win_y_high = y_current + window_height
        else: win_y_high = h

        # Check for [X] edge conditions
        if x_current - window_width >= 0: win_x_low = x_current - window_width
        else: win_x_low = 0

        if x_current + window_width <= w: win_x_high = x_current + window_width
        else: win_x_high = w

        cv2.circle(display_windows,(x_current,y_current),2,(255,0,255),-1)
        cv2.rectangle(display_windows,(win_x_low,win_y_high),(win_x_high,win_y_low),(255,255,0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

        if verbose == True:
            print("Current Window [" + str(window) + "] Center: " + str(x_current) + ", " + str(y_current))
            print("\tCurrent Window X Limits: " + str(win_x_low) + ", " + str(win_x_high))
            print("\tCurrent Window Y Limits: " + str(win_y_low) + ", " + str(win_y_high))
            print("\tCurrent Window # Good Pixels: " + str(len(good_inds)))

        pxl_count = len(good_inds)
        if pxl_count >= threshold:
            xmean = np.int(np.mean(nonzerox[good_inds]))
            ymean = np.int(np.mean(nonzeroy[good_inds]))

            mask_y_low = ymean - mask_size[0]
            mask_y_high = ymean + mask_size[0]
            mask_x_low = xmean - mask_size[1]
            mask_x_high = xmean + mask_size[1]

            x_subset.append(x_current)
            y_subset.append(y_current)

            x_current = xmean + window_width
            y_current = ymean - 2*window_height
            #         y_current = y_current - 2*window_height
            cv2.circle(display_windows,(xmean,ymean),2,(0,0,255),-1)
            cv2.rectangle(black,(mask_x_low,mask_y_high),(mask_x_high,mask_y_low),(255,255,255), cv2.FILLED)
        else:
            flag_done = True
            # x_current = x_current + window_width
            # x_current = np.int(np.mean(nonzerox[good_inds])) + window_width/2
        window += 1

    plot_image(display_windows,3)
    mask = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)
    mask_inv = cv2.bitwise_not(mask)
    return mask, mask_inv

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
        try:
            im = cv2.cvtColor(img,cspace)
        except:
            im = img
        imgs.append(im)

    helper_img_t = np.concatenate(
        (
            np.concatenate((imgs[0],sborder), axis=1),
            np.concatenate((imgs[1],sborder), axis=1),
            np.concatenate((imgs[2],sborder), axis=1)
        ), axis=1
    )

    helper_img_m = np.concatenate(
        (
            np.concatenate((bborder,cborder), axis=1),
            np.concatenate((bborder,cborder), axis=1),
            np.concatenate((bborder,cborder), axis=1)
        ), axis=1
    )

    helper_img_b = np.concatenate(
        (
            np.concatenate((imgs[3],sborder), axis=1),
            np.concatenate((imgs[4],sborder), axis=1),
            np.concatenate((imgs[5],sborder), axis=1)
        ), axis=1
    )

    helper_img = np.concatenate((helper_img_t,helper_img_m,helper_img_b), axis=0)
    return helper_img





def find_lines(_img="test/test_disparity.png", method = 1, line_input_method = 0,
    e1 = 2, e2 = 5, ang = 90, rho = 1, minLineLength = 11, maxLineGap = 11,
    houghThresh = 50, greyThresh = 11, cnt_thresh = 500.0, show_helpers = True,
    use_umap = True, flip_thresh_bin = False):

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
        if(not use_umap):
            _mask, mask_inv = window_slider(tmpIn,grey)
            plot_image(mask_inv,4)
            res = cv2.bitwise_and(tmp, tmp, mask = mask_inv)
            ret, grey = cv2.threshold(res,greyThresh,255,masking)

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

    try: hlines = cv2.cvtColor(hlines,cv2.COLOR_BGR2GRAY)
    except: pass
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
