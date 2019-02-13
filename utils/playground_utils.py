import os, sys, fnmatch
import cv2, time
import numpy as np
from matplotlib import pyplot as plt

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

        helper_imgs = [tmp,grey,dilation,blur,closing,closing]
        helper_img = construct_helper_img(helper_imgs)


    if(line_input_method is 1): hlines = blur
    elif(line_input_method is 2): hlines = blur
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
