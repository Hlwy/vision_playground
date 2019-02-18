import cv2, time
import numpy as np
from matplotlib import pyplot as plt

class UVMappingPipeline:
    def __init__(self):
        # Internally Stored Images
        self.img = []; self.normImg = []
        self.umap = []; self.vmap = []; self.overlay = []
        self.nonzero = []
        self.display = []
        self.filtered = []
        # Ground Segmentation Variables
        self.masked_ground = []
        self.ground_line = []
        self.ground_pts = []

        # Obstacle Information
        self.xBounds = []
        self.disparityBounds = []

        # Filtered Map Images
        self.greyU = []; self.closingU = []; self.dilatedU = []; self.blurredU = []
        self.greyV = []; self.closingV = []; self.dilatedV = []; self.blurredV = []

        # Other Useful Information
        self.contours = self.contoursU = self.contoursV = []
        self.contour_hierarchy = self.contour_hierarchyU = self.contour_hierarchyV = []

        # Previously Used Values (Helps prevent redundant computations)
        self.prev_img_path = None
        self.flag_new_img = False
        self.prev_map_flag = False
        self.prev_e1, self.prev_e2 = 0, 0
        self.prev_grey_threshU =  0
        self.prev_grey_threshV =  0

        # Sizes
        self.h = self.hv= 480
        self.w = self.wu = 640
        self.dmax = self.hu = self.wv = 256
        self.dmax_norm = 0
        self.window_size = [10,30]
        self.mask_size = [20,40]
        self.deadzone_vd = 4
        self.deadzone_ud = 3

        # Counters
        self.nPlots = 0

    """
    ============================================================================
        Main image processing pipeline (currently linked to iPyWidget events)
    ============================================================================
    """
    def pipeline(self, _img="test/test_disparity.png",
        line_method = 1, input_method = 0,
        e1 = 1, e2 = 5, ang = 90, rho = 1, minLineLength = 11, maxLineGap = 11,
        houghThresh = 50, greyThreshU = 35, greyThreshV=14, cnt_thresh = 50.0,
        show_helpers = True, use_umap = False, flip_thresh_bin = False
        ):

        # Parse inputs to create string descriptors for control variable copying
        if(input_method is 1): strFilterMeth = "Composite Filtering -> Blurring"
        elif(input_method is 2): strFilterMeth = "Composite Filtering -> Canny Edges"
        else: strFilterMeth = "Basic Thresholding"

        if(line_method is 0): strLineMeth = "Standard Hough Transform"
        else: strLineMeth = "Probablistic Hough Transform"

        if(use_umap): strMapSpace = "U-Map"
        else: strMapSpace = "V-Map"

        # Check variable arguments for changes to reduce repeating unnecessary calculations
        if((use_umap is not self.prev_map_flag) or (self.flag_new_img) or
           (greyThreshU is not self.prev_grey_threshU) or
           (greyThreshV is not self.prev_grey_threshV) or
           (e1 is not self.prev_e1) or (e2 is not self.prev_e2)
        ):
            flag_needs_filtering = True
            print("Images need to be filtered again...")
        else:
            flag_needs_filtering = False
            print("Skipping image filtering...")

        # Store current control paramters for next iteration
        self.prev_map_flag = use_umap
        self.prev_grey_threshU = greyThreshU
        self.prev_grey_threshV = greyThreshV
        self.prev_e1, self.prev_e2 = e1, e2

        print(
        """
        Inputs:  (image = %s)
        ------

        \t* Mapping Space               : %s
        \t* Line Finding Method         : %s
        \t* Filtering Used              : %s
        \t* Kernel Size                 : (%d, %d)
        \t* Rho, Angle (deg)            :  %d, %d
        \t* Min Line Length             :  %d
        \t* Max Line Gap                :  %d
        \t* Grey Thresholding (U-Map)   :  %d
        \t* Grey Thresholding (V-Map)   :  %d
        \t* Hough Threshold             :  %d
        \t* Contour Threshold           :  %d
        """ % (
            _img,strMapSpace,strLineMeth,strFilterMeth,e1,e2,rho,ang,minLineLength,maxLineGap,greyThreshU,greyThreshV,houghThresh,cnt_thresh
        ))
        # Convert Angle to radians for calculations
        ang = ang * np.pi/180

        # Begin Processing
        self.read_image(_img)
        self.get_uv_map()

        self.plot_image(self.overlay,0)

        self.filter_image(greyThreshU, greyThreshV, e1, e2, timing=True)
        if(use_umap): cntImgU = self.greyU
        else: cntImgU = self.dilatedU

        # cntsV, hierV = self.find_contours(self.greyV,cnt_thresh)
        # self.contoursV = cntsV; self.contour_hierarchyV = hierV
        cntsU, hierU = self.find_contours(cntImgU,cnt_thresh)
        self.contoursU = cntsU; self.contour_hierarchyU = hierU

        _cnts, limits, ellipses, pts, centers = self.extract_contour_x_bounds()

        self.nObs = len(self.disparityBounds)
        self.obsImgs = []
        disp = np.copy(self.img)
        self.display_obstacles = cv2.cvtColor(disp,cv2.COLOR_GRAY2BGR)
        for i in range(self.nObs):
            tmp = np.copy(self.img)
            tmp = cv2.cvtColor(tmp,cv2.COLOR_GRAY2BGR)
            try:
                xs = self.xBounds[i]
                ys,disp = self.find_obstacles(self.greyV,self.disparityBounds[i], _threshold=30, lower_threshold=1, verbose = True, display=self.greyV)
                cv2.rectangle(tmp,(xs[0],ys[0]),(xs[1],ys[-1]),(150,0,0),1)
                cv2.rectangle(self.display_obstacles,(xs[0],ys[0]),(xs[1],ys[-1]),(150,0,0),1)
            except: pass
            self.obsImgs.append(tmp)

        self.plot_image(self.display_obstacles,2)

        if(input_method is 1):
            inputU = self.dilatedU
            inputV = self.dilatedV
        elif(input_method is 2):
            inputU = self.blurredU
            inputV = self.blurredV
        else:
            inputU = self.greyU
            inputV = self.greyV

        dispU = self.line_finder(inputU, rho, ang, houghThresh, minLineLength, maxLineGap, line_method)
        dispV = self.line_finder(inputV, rho, ang, houghThresh, minLineLength, maxLineGap, line_method)

        # comp_imgs = [self.vmap,self.greyV,self.dilatedV,self.blurredV,self.closingV,dispV]
        comp_imgs = [self.umap,self.greyU,self.dilatedU,self.blurredU,self.closingU,dispU]
        composite = self.construct_composite_img(comp_imgs,2,3)
        self.plot_image(composite,3)

    """
    ============================================================================
        Attempt to find the y boundaries for potential obstacles found from
        the U-Map
    ============================================================================
    """
    def find_obstacles(self, _img, disparityRange, _threshold=30, lower_threshold=1, draw_windows=True, verbose=False, timing=False, display=None):
        if(timing): t0 = time.time()
        # Temporary dump variables
        count = 0; winCount = 0
        maxStep = 40
        flag_done = False
        flag_found_start = False
        good_inds = [] # Indices used for keeping track of pixels w/in current window
        mean_pxls = []
        ys = []

        h,w = _img.shape[:2]

        try: imgV = cv2.cvtColor(_img,cv2.COLOR_GRAY2BGR)
        except: imgV = np.copy(_img)
        # Keep a copy of the original input image for overlaying windows
        try:
            d = display.dtype
            display_windows = np.copy(display)
        except:
            display_windows = np.copy(imgV)

        tmp = np.copy(imgV)

        # Prevent initial search coordinates from clipping search window at edges
        xk = (disparityRange[1] + disparityRange[0]) / 2
        yk = 0
        dWy = 20
        dWx = abs(xk - disparityRange[0])
        if(dWx <= 0): dWx = 1

        if(yk <= 0): yk = 0 + dWy
        if(verbose): print("Starting Location: ", xk, yk, dWx)

        # Grab all nonzero pixels
        nonzero = tmp.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Begin Sliding Window Search Technique
        while(winCount <= maxStep and not flag_done):
            # Store previously found indices as a fallback in case next window is insufficient
            prev_good_inds = good_inds

            # If we are at image width we must stop
            if(xk >= w):
                flag_done = True
                if(verbose): print("Exiting: Reached max image width.")

            # Slide window from previousy found center (Clip at image edges)
            # Update vertical [Y] window edges
            if(yk - dWy >= 0): wy_low = yk - dWy
            else: wy_low = 0

            if(yk + dWy <= h): wy_high = yk + dWy
            else: wy_high = h

            # Update horizontal [X] window edges
            if(xk - dWx >= 0): wx_low = xk - dWx
            else: wx_low = 0

            if(xk + dWx <= w): wx_high = xk + dWx
            else: wx_high = w

            if(draw_windows):
                cv2.circle(display_windows,(xk,yk),2,(255,0,255),-1)
                cv2.rectangle(display_windows,(wx_low,wy_high),(wx_high,wy_low),(255,255,0), 1)

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= wy_low) & (nonzeroy < wy_high) &
                        (nonzerox >= wx_low) &  (nonzerox < wx_high)).nonzero()[0]
            nPxls = len(good_inds)
            if verbose == True:
                print("Current Window [" + str(count) + "] Center: " + str(xk) + ", " + str(yk))
                # print("\tCurrent Window X Limits: " + str(wx_low) + ", " + str(wx_high))
                # print("\tCurrent Window Y Limits: " + str(wy_low) + ", " + str(wy_high))
                print("\tCurrent Window # Good Pixels: " + str(nPxls))

            # Record mean coordinates of pixels in window and update new window center
            if(nPxls >= _threshold):
                xmean = np.int(np.mean(nonzerox[good_inds]))
                ymean = np.int(np.mean(nonzeroy[good_inds]))
                mean_pxls.append([xmean,ymean])
                if(count == 0): ys.append(yk - dWy)
                else: ys.append(yk)
                flag_found_start = True
                count+=1
                if(draw_windows): cv2.circle(display_windows,(xmean,ymean),2,(0,0,255),-1)
            elif(nPxls <= lower_threshold and flag_found_start):
                flag_done = True
                ys.append(yk)

            # Update New window center coordinates
            xk = xk
            yk = yk + 2*dWy
            winCount += 1

        if(draw_windows): self.plot_image(display_windows,5)

        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[get_vmap_mask] --- Took %f seconds to complete" % (dt))
        return ys, display_windows


    """
    ============================================================================
        Pre-process the UV-Maps before being used to detect obstacles
    ============================================================================
    """
    def filter_image(self, grey_thresholdU, grey_thresholdV, _e1, _e2, verbose=False, timing=False):
        if(timing): t0 = time.time()
        # Morphological Kernels
        kernelU = np.ones((_e1,_e2),np.uint8)
        kernelV = np.ones((_e2,_e1),np.uint8)

        # U-Map: Disparity Filtering
        tmpU = np.copy(self.umap)
        tmpU = cv2.cvtColor(tmpU,cv2.COLOR_GRAY2BGR)
        # Filter out disparities extremely close
        tmpH,tmpW = tmpU.shape[:2]
        black = np.zeros((tmpH,tmpW,3),dtype=np.uint8)
        cv2.rectangle(black,(0,0),(tmpW,self.deadzone_ud),(255,255,255), cv2.FILLED)
        cv2.rectangle(black,(0,tmpH-self.deadzone_ud),(tmpW,tmpH),(255,255,255), cv2.FILLED)
        maskU = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)
        maskU_inv = cv2.bitwise_not(maskU)
        resU = cv2.bitwise_and(tmpU, tmpU, mask = maskU_inv)

        _, greyU = cv2.threshold(resU,grey_thresholdU,255,cv2.THRESH_BINARY)

        dilation_U = cv2.dilate(greyU,kernelU,iterations = 1)
        blur_U = cv2.GaussianBlur(dilation_U,(5,5),0)
        closing_U = cv2.morphologyEx(greyU,cv2.MORPH_CLOSE,kernelU, iterations = 2)

        # V-Map: Disparity Filtering w/ siliding window filter
        tmpV = np.copy(self.vmap)
        tmpV = cv2.cvtColor(tmpV,cv2.COLOR_GRAY2BGR)
        _, greyV = cv2.threshold(tmpV,grey_thresholdV,255,cv2.THRESH_BINARY)

        # Sliding Window Technique for filtering out ground
        maskV, invMaskV = self.get_vmap_mask(greyV,verbose=False)
        resV = cv2.bitwise_and(tmpV, tmpV, mask = invMaskV)
        _, greyV = cv2.threshold(resV,grey_thresholdV,255,cv2.THRESH_BINARY)

        dilation_V = cv2.dilate(greyV,kernelV,iterations = 1)
        blur_V = cv2.GaussianBlur(dilation_V,(5,5),0)
        closing_V = cv2.morphologyEx(greyV,cv2.MORPH_CLOSE,kernelV, iterations = 2)

        # Store Individually filtered images internally for external testing
        self.greyU = np.copy(greyU)
        self.dilatedU = np.copy(dilation_U)
        self.blurredU = np.copy(blur_U)
        self.closingU = np.copy(closing_U)

        self.vMasked = np.copy(resV)
        self.greyV = np.copy(greyV)
        self.dilatedV = np.copy(dilation_V)
        self.blurredV = np.copy(blur_V)
        self.closingV = np.copy(closing_V)
        self.masked_ground = cv2.bitwise_and(tmpV, tmpV, mask = maskV)

        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[filter_image] --- Took %f seconds to complete" % (dt))

        return 0

    """
    ============================================================================
        Attempt to find all the contours from input image w/ area filtering
    ============================================================================
    """
    def find_contours(self, _img, _threshold=0, verbose=True, timing=True, use_areas=False):
        areas = []; perimeters = []
        filtered_areas = []; filtered_perimeters = []
        if(timing): t0 = time.time()

        try: img = cv2.cvtColor(_img,cv2.COLOR_BGR2GRAY)
        except: img = np.copy(_img)
        display = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        _, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(cnt) for cnt in contours]
        perimeters = [cv2.arcLength(cnt,True) for cnt in contours]

        if(use_areas):
            filtered_cnts = [cnt for cnt in contours if cv2.contourArea(cnt) >= _threshold]
            filtered_areas = [cv2.contourArea(cnt) for cnt in filtered_cnts]
        else:
            filtered_cnts = [cnt for cnt in contours if cv2.arcLength(cnt,True) >= _threshold]
            filtered_perimeters = [cv2.arcLength(cnt,True) for cnt in filtered_cnts]

        self.contours = filtered_cnts
        self.contour_hierarchy = hierarchy

        if(verbose):
            if(use_areas):
                print("Raw Contour Areas:",areas)
                print("Filtered Contour Areas:",filtered_areas)
            else:
                print("Raw Contour Perimeters:",perimeters)
                print("Filtered Contour Perimeters:",filtered_perimeters)

        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[find_contours] --- Took %f seconds to complete" % (dt))

        return filtered_cnts, hierarchy

    """
    ============================================================================
        Attempt to find the horizontal bounds for detected contours
    ============================================================================
    """
    def extract_contour_x_bounds(self, _cnts=None, _display=None, verbose=False, timing=True):
        if(_display is None): display = np.copy(self.greyU)
        else: display = np.copy(_display)

        if(_cnts is None): cnts = self.contoursU
        else: cnts = _cnts

        limits = []; ellipses = []; pts = []; centers = []
        self.xBounds = []
        self.disparityBounds = []

        # print(x, x+w, y, y+h)

        if(timing): t0 = time.time()

        for cnt in cnts:
            try:
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
            except: pass

            try:
                x,y,rectw,recth = cv2.boundingRect(cnt)
                self.xBounds.append([x, x + rectw])
                self.disparityBounds.append([y, y + recth])
            except: pass

        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[extract_contour_x_bounds] --- Took %f seconds to complete" % (dt))

        self.plot_contours(display,cnts,ellipses,centers,pts)
        return cnts, limits, ellipses, pts, centers

    """
    ============================================================================
        Overlay contours and other relative information onto an image
    ============================================================================
    """
    def plot_contours(self, _img, _cnts, ellipses, centers, pts):

        try: img = cv2.cvtColor(_img,cv2.COLOR_GRAY2BGR)
        except: img = np.copy(_img)
        try: tmp = cv2.cvtColor(_img,cv2.COLOR_GRAY2BGR)
        except: tmp = np.copy(_img)
        display = np.copy(tmp)

        for i, cnt in enumerate(_cnts):
            cv2.drawContours(tmp, [cnt], 0, (255,0,0), 2)
            try:
                cv2.ellipse(display,ellipses[i],(0,255,0),2)
                cv2.circle(display,centers[i],2,(255,0,255), 5)
                cv2.circle(display,pts[i][0],2,(0,255,255), 5)
                cv2.circle(display,pts[i][1],2,(0,255,255), 5)
            except: pass
        sborder = np.ones((_img.shape[0],5,3),dtype=np.uint8)
        sborder[np.where((sborder==[1,1,1]).all(axis=2))] = [255,0,255]
        helper = np.concatenate((
            np.concatenate((img,sborder), axis=1),
            np.concatenate((tmp,sborder), axis=1),
            np.concatenate((display,sborder), axis=1)
        ), axis=1)
        self.plot_image(helper,4)

    """
    ============================================================================
        Abstract a mask image for filtering out the ground from a V-Map
    ============================================================================
    """
    def get_vmap_mask(self, _img, _threshold=30, plot_histogram=False, draw_windows=True, verbose=False, timing=False):
        if(timing): t0 = time.time()
        # Temporary dump variables
        display = None
        count = 0
        maxStep = 40
        deadzone_x = self.deadzone_vd
        flag_done = False
        good_inds = [] # Indices used for keeping track of pixels w/in current window
        mean_pxls = []

        dWy,dWx = np.int32(self.window_size)/2
        dMy,dMx = np.int32(self.mask_size)/2

        h,w = _img.shape[:2]
        # Create a black template to create mask with
        black = np.zeros((h,w,3),dtype=np.uint8)

        try: imgV = cv2.cvtColor(_img,cv2.COLOR_GRAY2BGR)
        except: imgV = np.copy(_img)
        # Keep a copy of the original input image for overlaying windows
        try:
            d = display.dtype
            display_windows = np.copy(display)
        except:
            display_windows = np.copy(imgV)

        tmp = np.copy(imgV)

        # Take the bottom strip of the input image to find potential points to start sliding from
        hist = np.sum(tmp[h-dWy:h,deadzone_x:w], axis=0)
        x0 = abs(int(np.argmax(hist[:,0])))
        y0 = abs(int(h))
        # Prevent initial search coordinates from clipping search window at edges
        if(x0 <= dWx): xk = dWx
        else: xk = x0

        if(y0 >= h): yk = h - dWy
        else: yk = y0
        if(verbose): print("Starting Location: ", xk, yk)

        if(plot_histogram):
            plt.figure(self.nPlot+1)
            plt.plot(range(hist.shape[0]),hist[:])
            plt.show()

        # Grab all nonzero pixels
        nonzero = tmp.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Begin Sliding Window Search Technique
        while(count <= maxStep and not flag_done):
            # Store previously found indices as a fallback in case next window is insufficient
            prev_good_inds = good_inds

            # TODO: Modify search window width depending on the current disparity
            if(xk >= w/2): dWx = int(self.window_size[1])
            # If we are at image width we must stop
            elif(xk >= w):
                flag_done = True
                if(verbose): print("Exiting: Reached max image width.")

            # Slide window from previousy found center (Clip at image edges)
            # Update vertical [Y] window edges
            if(yk - dWy >= 0): wy_low = yk - dWy
            else: wy_low = 0

            if(yk + dWy <= h): wy_high = yk + dWy
            else: wy_high = h

            # Update horizontal [X] window edges
            if(xk - dWx >= 0): wx_low = xk - dWx
            else: wx_low = 0

            if(xk + dWx <= w): wx_high = xk + dWx
            else: wx_high = w

            if(draw_windows):
                cv2.circle(display_windows,(xk,yk),2,(255,0,255),-1)
                cv2.rectangle(display_windows,(wx_low,wy_high),(wx_high,wy_low),(255,255,0), 1)

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= wy_low) & (nonzeroy < wy_high) &
                        (nonzerox >= wx_low) &  (nonzerox < wx_high)).nonzero()[0]
            nPxls = len(good_inds)
            if verbose == True:
                print("Current Window [" + str(count) + "] Center: " + str(xk) + ", " + str(yk))
                print("\tCurrent Window X Limits: " + str(wx_low) + ", " + str(wx_high))
                print("\tCurrent Window Y Limits: " + str(wy_low) + ", " + str(wy_high))
                print("\tCurrent Window # Good Pixels: " + str(nPxls))

            # Record mean coordinates of pixels in window and update new window center
            if(nPxls >= _threshold):
                xmean = np.int(np.mean(nonzerox[good_inds]))
                ymean = np.int(np.mean(nonzeroy[good_inds]))
                mean_pxls.append([xmean,ymean])

                # Draw current window onto mask template
                my_low = ymean - dMy
                my_high = ymean + dMy
                mx_low = xmean - dMx
                mx_high = xmean + dMx
                cv2.rectangle(black,(mx_low,my_high),(mx_high,my_low),(255,255,255), cv2.FILLED)

                # Update New window center coordinates
                xk = xmean + dWx
                yk = ymean - 2*dWy

                if(draw_windows): cv2.circle(display_windows,(xmean,ymean),2,(0,0,255),-1)
            else: flag_done = True
            count += 1

        if(draw_windows): self.plot_image(display_windows,1)

        cv2.rectangle(black,(0,0),(deadzone_x,h),(255,255,255), cv2.FILLED)
        mask = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(mask)

        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[get_vmap_mask] --- Took %f seconds to complete" % (dt))

        return mask, mask_inv

    """
    ============================================================================
        Create the UV Disparity Mappings from a given depth (disparity) image
    ============================================================================
    """
    def get_uv_map(self, _img=None, get_overlay=True, verbose=True, timing=False):
        overlay = []
        if(_img is None):
            img = np.copy(self.img)
            dmax = np.max(img) + 1
        else:
            img = np.copy(_img)
            dmax = np.max(img)

        # Determine stats for U and V map images
        h, w = img.shape[:2]
        hu, wu = dmax, w
        hv, wv = h, dmax

        histRange = (0,dmax)
        if(verbose): print("[UV Mapping] Input Image Size: (%d, %d) --- w/ max disparity = %.3f" % (h,w, dmax))

        umap = np.zeros((hu,wu,1), dtype=np.uint8)
        vmap = np.zeros((hv,wv,1), dtype=np.uint8)

        if(timing): t0 = time.time()

        for i in range(0,h):
            vscan = img[i,:]
            vrow = cv2.calcHist([vscan],[0],None,[dmax],histRange)
            if(verbose): print("\t[V Mapping] Scan [%d] (%s) ---- Scan Histogram (%s)" % (i,', '.join(map(str, vscan.shape)), ', '.join(map(str, vrow.shape))))
            vmap[i,:] = vrow

        for i in range(0,w):
            uscan = img[:,i]
            urow = cv2.calcHist([uscan],[0],None,[dmax],histRange)
            if(verbose): print("\t[U Mapping] Scan[%d] (%s) ---- Scan Histogram (%s)" % (i,', '.join(map(str, uscan.shape)), ', '.join(map(str, urow.shape))))
            umap[:,i] = urow

        umap = np.reshape(umap,(hu,wu))
        vmap = np.reshape(vmap,(hv,wv))

        if(get_overlay):
            blank = np.ones((umap.shape[0],vmap.shape[1]),np.uint8)*255
            pt1 = np.concatenate((img, vmap), axis=1)
            pt2 = np.concatenate((umap,blank),axis=1)
            overlay = np.concatenate((pt1,pt2),axis=0)
            self.overlay = cv2.cvtColor(overlay,cv2.COLOR_GRAY2BGR)

        if(verbose): print("\t[UV Mapping] U Map = (%s) ----- V Map = (%s)" % (', '.join(map(str, umap.shape)),', '.join(map(str, vmap.shape)) ))
        self.umap = np.copy(umap)
        self.vmap = np.copy(vmap)
        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[UV Mapping] --- Took %f seconds to complete" % (dt))
        return umap,vmap,overlay

    """
    ============================================================================
        Attempt to find lines from input image using Hough Transformation
    ============================================================================
    """
    def line_finder(self, _img, rho, angle, _threshold, _length, _gap, _method=0):
        lines = []
        try: img = cv2.cvtColor(_img,cv2.COLOR_BGR2GRAY)
        except: img = np.copy(_img)
        display = np.copy(img)

        try:
            if(_method==0): # Find lines using standard Hough Transformation
                count = 0
                lines = cv2.HoughLines(img,rho,angle,_threshold)
                for rho,theta in lines[0]:
                    count+=1
                    a = np.cos(theta); b = np.sin(theta)
                    x0 = a*rho; y0 = b*rho
                    x1 = int(x0 + 1000*(-b)); y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b)); y2 = int(y0 - 1000*(a))

                    cv2.line(display,(x1,y1),(x2,y2),(0,0,255),2)
                    # cv2.line(filtered,(x1,y1),(x2,y2),(255,255,255),2)
            else:
                lines = cv2.HoughLinesP(img,rho,angle,_threshold,_length,_gap)
                for x in range(0, len(lines)):
                    for x1,y1,x2,y2 in lines[x]:
                        cv2.line(display,(x1,y1),(x2,y2),(0,255,0),2)
                        # cv2.line(filtered,(x1,y1),(x2,y2),(255,255,255),2)
        except:
            print("Couldn't Find Hough Lines!!!")
            pass
        return display

    """
    ============================================================================
        Attempt to fit a polynomial line to the segmented out ground V-Map
    ============================================================================
    """
    def fit_ground_line(self, _vmap, degree=3):
        tmp = np.copy(_vmap)
        try: img = cv2.cvtColor(tmp,cv2.COLOR_GRAY2BGR)
        except: img = tmp
        h,w = img.shape[0], img.shape[1]

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        ploty = np.linspace(0, w)

        try: # Try fitting polynomial nonzero data
            fit = np.poly1d(np.polyfit(nonzerox,nonzeroy, degree))
            # Generate x and y values for plotting
            plotx = fit(ploty)
        except:
            print("ERROR: Function 'polyfit' failed for LEFT SIDE!")
            fit = [0, 0]; plotx = [0, 0]

        if(verbose): print(fit)

        xs = np.asarray(ploty,dtype=np.int32); ys = np.asarray(plotx,dtype=np.int32)
        pts = np.vstack(([xs],[ys])).transpose()
        cv2.polylines(img, [pts], 0, (255,0,0))

        plt.figure(4); plt.clf(); plt.imshow(img); plt.show()
        self.ground_line = fit
        self.ground_pts = pts
        return fit

    """ ============================================================
        Load a depth image from the specified path only if the given path
        is different from the previously used path
        ============================================================
    """
    def read_image(self,_img):
        if(_img != self.prev_img_path):
            self.img = cv2.imread(_img,cv2.IMREAD_GRAYSCALE)
            self.normImg = (self.img - np.mean(self.img))/np.std(self.img)
            # Get stats on original image
            self.h, self.w = self.img.shape[:2]
            self.dmax = np.max(self.img) + 1
            self.dmax_norm = np.max(self.normImg)
            # Misc
            self.prev_img_path = _img
            self.flag_new_img = True
        else:
            self.flag_new_img = False

    """
    ============================================================================
                                Plot an Image
    ============================================================================
    """
    def plot_image(self,img,figNum=None):
        if(figNum == None): plt.figure()
        else:
            plt.figure(figNum)
            self.nPlots += 1

        plt.imshow(img)
        plt.subplots_adjust(wspace=0.0,hspace=0.0,left=0.0,right=1.0,top=1.0, bottom=0.0)
        plt.show()

    """
    ============================================================================
        Stitch together a list of images into a composited image for easy viewing
    ============================================================================
    """
    def construct_composite_img(self,_imgs,nRows,nCols, border_width=5, color=(255,0,255) ):
        n,m = _imgs[0].shape[0], _imgs[0].shape[1]
        # Create Image Borders for visual seperation
        border_bot = np.ones((border_width,m,3),dtype=np.uint8)
        border_side = np.ones((n,border_width,3),dtype=np.uint8)
        border_corner = np.ones((border_width,border_width,3),dtype=np.uint8)
        # Fill Border images with color
        border_bot[:] = color; border_side[:] = color; border_corner[:] = color

        # Attempt to convert all input images into correct colorspace
        imgs = []
        for _img in _imgs:
            try: img = cv2.cvtColor(_img,cv2.COLOR_GRAY2BGR)
            except: img = np.copy(_img)
            imgs.append(img)

        # Assemble composite image
        count = 0
        sections = []
        for row in range(nRows):
            tiles = []; tileBorders = []
            for col in range(nCols):
                tmp = np.concatenate((imgs[count],border_side), axis=1)
                tmpBorder = np.concatenate((border_bot,border_corner), axis=1)
                tiles.append(tmp); tileBorders.append(tmpBorder)
                count+=1
            sections.append(np.concatenate(tiles, axis=1))
            sections.append(np.concatenate(tileBorders, axis=1))

        return np.concatenate(sections, axis=0)


    def modify_image_by_pixels(self,_img, condition=255):
        tmp = np.copy(_img)
        indices = np.argwhere(tmp == condition)

        indices = indices[:,:2]
        disparity_filters = indices[:,0]
        tmp[np.isin(tmp, disparity_filters)] = 0
