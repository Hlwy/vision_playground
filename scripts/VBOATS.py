import numpy as np
import os, sys, cv2, time, math
from matplotlib import pyplot as plt

# sys.path.append(os.path.abspath(os.path.join('..', '')))

from hyutils.img_utils import *
from hyutils.debug_utils import *

class VBOATS:
    def __init__(self):
        # Internally Stored Images
        self.img = []

        # Other Useful Information
        self.umap_raw = []
        self.umap_deadzoned = []
        self.umap_processed = []
        self.stripsU_raw = []
        self.stripsU_processed = []

        self.vmask = []
        self.vmap_raw = []
        self.vmap_deadzoned = []
        self.vmap_filtered = []
        self.vmap_processed = []
        self.stripsV_raw = []
        self.stripsV_processed = []

        self.xBounds = []
        self.yBounds = []
        self.dbounds = []
        self.disparityBounds = []

        self.filtered_contours = []
        self.obstacles = []
        self.obstacles_umap = []
        self.ground_pxls = []

        self.windows_obstacles = []
        self.windows_ground = []

        # Sizes
        self.h = 480
        self.w = 640
        self.dmax = 256
        self.window_size = [10,30]
        self.vmap_search_window_height = 10
        self.mask_size = [20,40]
        self.dead_x = 10
        self.dead_y = 3
        self.dH = 4
        self.dW = 5
        self.dThs = [0.4, 0.4,0.1, 0.15]

        self.deadThreshMin = 85
        self.dHplus = 10
        self.kszs = [
            (10,2),
            (50,5),
            (75,1)
        ]
        self.deadUThresh = 0.8
        self.testLowCntThresh = 35.0
        self.testHighCntThresh = 50.0
        self.testRestThreshRatio = 0.25
        self.testMaxGndStep = 16
        self.is_ground_present = True
        # Counters
        self.nObs = 0
        # Flags
        self.flag_simulation = False
        self.flag_morphs = True
        self.debug = False
        self.debug_contours = False
        self.debug_windows = False
        self.debug_obstacle_search = False
    def get_uv_map(self, img, verbose=False, timing=False):
        """ ===================================================================
        Create the UV Disparity Mappings from a given depth (disparity) image
        ==================================================================== """
        dt = 0
        if(timing): t0 = time.time()
        dmax = np.uint8(np.max(img)) + 1

        # Determine stats for U and V map images
        h, w = img.shape[:2]
        hu, wu = dmax, w
        hv, wv = h, dmax
        histRange = (0,dmax)

        if(verbose):
            print("[UV Mapping] Input Image Size: (%d, %d) --- w/ max disparity = %.3f" % (h,w, dmax))
            print("[UV Mapping] Disparity Map Sizes --- U-Map (%.2f, %.2f) ||  V-Map (%.2f, %.2f)" % (hu, wu, hv, wv))

        umap = np.zeros((dmax,w,1), dtype=np.uint8)
        vmap = np.zeros((h,dmax,1), dtype=np.uint8)

        for i in range(0,w):
            uscan = img[:,i]
            urow = cv2.calcHist([uscan],[0],None,[dmax],histRange)
            if(verbose): print("\t[U Mapping] Scan[%d] (%s) ---- Scan Histogram (%s)" % (i,', '.join(map(str, uscan.shape)), ', '.join(map(str, urow.shape))))
            umap[:,i] = urow

        for i in range(0,h):
            vscan = img[i,:]
            vrow = cv2.calcHist([vscan],[0],None,[dmax],histRange)
            if(verbose): print("\t[V Mapping] Scan [%d] (%s) ---- Scan Histogram (%s)" % (i,', '.join(map(str, vscan.shape)), ', '.join(map(str, vrow.shape))))
            vmap[i,:] = vrow

        umap = np.reshape(umap,(dmax,w))
        vmap = np.reshape(vmap,(h,dmax))

        if(verbose): print("\t[UV Mapping] U Map = (%s) ----- V Map = (%s)" % (', '.join(map(str, umap.shape)),', '.join(map(str, vmap.shape)) ))

        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[UV Mapping] --- Took %f seconds to complete" % (dt))

        return umap,vmap, dt
    def extract_contour_bounds(self, cnts, verbose=False, timing=False):
        """ ===================================================================
        	Attempt to find the horizontal bounds for detected contours
        ==================================================================== """
        dt = 0
        xBounds = []
        disparityBounds = []

        if(timing): t0 = time.time()

        for cnt in cnts:
            try:
                x,y,rectw,recth = cv2.boundingRect(cnt)
                xBounds.append([x, x + rectw])
                disparityBounds.append([y, y + recth])
            except: pass

        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[extract_contour_bounds] --- Took %f seconds to complete" % (dt))

        return xBounds, disparityBounds, dt
    def find_contours(self, _umap, threshold = 30.0, threshold_method = "perimeter", offset=(0,0), max_thresh=1500.0, debug=False):
        """
        ============================================================================
        	Find contours in image and filter out those above a certain threshold
        ============================================================================
        """
        umap = np.copy(_umap)
        # try: umap = cv2.cvtColor(_umap,cv2.COLOR_BGR2GRAY)
        # except: print("[WARNING] find_contours --- Unnecessary Image Color Converting")

        _, contours, hierarchy = cv2.findContours(umap,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE,offset=offset)

        if(threshold_method == "perimeter"):
            filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt,True) >= threshold]
            filtered_contours = [cnt for cnt in filtered_contours if cv2.arcLength(cnt,True) <= max_thresh]
            if(debug):
                raw_perimeters = np.array([cv2.arcLength(cnt,True) for cnt in contours])
                filtered_perimeters = np.array([cv2.arcLength(cnt,True) for cnt in filtered_contours])
                print("Raw Contour Perimeters:",np.unique(raw_perimeters))
                print("Filtered Contour Perimeters:",np.unique(filtered_perimeters))
        elif(threshold_method == "area"):
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= threshold]
            if(debug):
                raw_areas = np.array([cv2.contourArea(cnt) for cnt in contours])
                filtered_areas = np.array([cv2.contourArea(cnt) for cnt in filtered_contours])
                print("Raw Contour Areas:",np.unique(raw_areas))
                print("Filtered Contour Areas:",np.unique(filtered_areas))
        else:
            print("[ERROR] find_contours --- Unsupported filtering method!")

        return filtered_contours,contours
    def find_obstacles(self, vmap, dLims, xLims, search_thresholds = (3,30), ground_detected=True, verbose=False):
        """
        ============================================================================
                        Find obstacles within a given V-Map
        ============================================================================
        """
        obs = []; obsUmap = []; windows = []; ybounds = []; dBounds = []
        nObs = len(dLims)
        for i in range(nObs):
            xs = xLims[i]
            ds = dLims[i]
            ys,ws,_ = self.obstacle_search(vmap, ds, search_thresholds,verbose=verbose)
            if self.debug_obstacle_search: print("Y Limits[",len(ys),"]: ", ys)
            if(len(ys) <= 2 and ground_detected):
                if(verbose): print("[INFO] Found obstacle with zero height. Skipping...")
            elif(len(ys) <= 1):
                if(verbose): print("[INFO] Found obstacle with zero height. Skipping...")
            elif(ys[0] == ys[1]):
                if(verbose): print("[INFO] Found obstacle with zero height. Skipping...")
            else:
                ybounds.append(ys)
                obs.append([
                    (xs[0],ys[0]),
                    (xs[1],ys[-1])
                ])
                obsUmap.append([
                    (xs[0],ds[0]),
                    (xs[1],ds[1])
                ])
                windows.append(ws)
                dBounds.append(ds)
        return obs, obsUmap, ybounds, dBounds, windows, len(obs)
    def obstacle_search(self, _vmap, x_limits, pixel_thresholds=(1,30), window_size=None, verbose=False, timing=False):
        """
        ============================================================================
        	Attempt to find the y boundaries for potential obstacles found from
        	the U-Map
        ============================================================================
        """
        flag_done = False
        try_last_search = False
        tried_last_resort = False
        flag_found_start = False
        flag_hit_limits = False
        count = 0; nWindows = 0; dt = 0
        yLims = []; windows = []

        try: img = cv2.cvtColor(_vmap,cv2.COLOR_GRAY2BGR)
        except:
            img = np.copy(_vmap)
            print("[WARNING] obstacle_search ------------  Unnecessary Image Color Converting")
        # Limits
        h,w = img.shape[:2]
        xmin, xmax = x_limits
        pxlMin, pxlMax = pixel_thresholds
        # Starting Pixel Coordinate
        yk = prev_yk = 0
        xk = (xmax + xmin) / 2
        # Window Size
        if(window_size is None):
            dWy = self.vmap_search_window_height
            dWx = abs(xk - xmin)
            if(dWx <= 1):
                if xk <= 3: dWx = 1
                else:  dWx = 2
        else: dWx, dWy = np.int32(window_size)/2

        if(yk <= 0): yk = 0 + dWy
        if(verbose): print("Object Search Window [%d x %d] -- Starting Location (%d, %d)" % (dWx,dWy,xk, yk) )

        if(timing): t0 = time.time()

        # Grab all nonzero pixels
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Begin Sliding Window Search Technique
        while(not flag_done):
            if(xk >= w): # If we are at image edges we must stop
                flag_hit_limits = True
                if(verbose): print("[INFO] Reached max image width.")
            if(yk + dWy >= h):
                flag_hit_limits = True
                if(verbose): print("[INFO] Reached max image height.")
            if(flag_hit_limits): flag_done = True

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

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= wy_low) & (nonzeroy < wy_high) &
                        (nonzerox >= wx_low) &  (nonzerox < wx_high)).nonzero()[0]
            nPxls = len(good_inds)
            if(verbose):
                print("Current Window [" + str(count) + "] ----- Center: " + str(xk) + ", " + str(yk) + " ----- # of good pixels = "  + str(nPxls))

            # Record mean coordinates of pixels in window and update new window center
            if(nPxls >= pxlMax):
                if(nWindows == 0):
                    yLims.append(yk - dWy)
                    if flag_hit_limits:
                        flag_done = False
                        print("\tTrying Last Ditch Search...")
                else:
                    yLims.append(yk)
                    try_last_search = False
                windows.append([(wx_low,wy_high),(wx_high,wy_low)])
                nWindows += 1
                prev_yk = yk
                flag_found_start = True
            elif(nPxls <= pxlMin and flag_found_start):
                flag_done = True
                yLims.append(prev_yk + dWy)
            # Update New window center coordinates
            xk = xk
            yk = yk + 2*dWy
            count += 1

        if(verbose): print("[%d] Good Windows" % nWindows)
        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[obstacle_search] --- Took %f seconds to complete" % (dt))

        return yLims, windows, dt
    def get_vmap_mask(self, vmap, threshold=20, min_ground_pixels=6, dxmean_thresh=1.0, maxStep=14, deltas=(0,20), mask_size = [10,30], window_size = [10,30], draw_method=1, verbose=False, timing=False):
        """
        ============================================================================
        	Abstract a mask image for filtering out the ground from a V-Map
        ============================================================================
        """
        flag_done = False
        count = 0; dt = 0
        prev_xmean = mean_count = mean_sum = 0

        good_inds = []; mean_pxls = []; windows = []; masks = []
        # Sizes
        h,w = vmap.shape[:2]
        dWy,dWx = np.int32(window_size)/2
        dWyO,dWxO = np.int32(window_size)/2
        dMy,dMx = np.int32(mask_size)/2
        dx,dy = np.int32(deltas)
        # ==========================================================================

        # Create a black template to create mask with
        black = np.zeros((h,w,3),dtype=np.uint8)

        if(timing): t0 = time.time()
        # ==========================================================================

        if(verbose): print("[INFO] get_vmap_mask() ---- vmap shape (H,W,C): %s" % (str(vmap.shape)))
        # Take the bottom strip of the input image to find potential points to start sliding from
        y0 = abs(int(h-dy))
        hist = np.sum(vmap[y0:h,0:w], axis=0)
        if(verbose): print("[INFO] get_vmap_mask() ---- hist shape (H,W,C): %s" % (str(hist.shape)))
        try:
            x0 = abs(int(np.argmax(hist[:,0])))
        except:
            x0 = abs(int(np.argmax(hist[:])))

        # ==========================================================================

        # Prevent initial search coordinates from clipping search window at edges
        if(x0 <= dWx): xk = dWx
        else: xk = x0
        if(y0 >= h): yk = h - dWy
        else: yk = y0
        if(verbose): print("[get_vmap_mask] --- Starting Location: ", xk, yk)
        # ==========================================================================

        # Grab all nonzero pixels
        nonzero = vmap.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Begin Sliding Window Search Technique
        while(count <= maxStep and not flag_done):
            # TODO: Modify search window width depending on the current disparity
            if(xk >= w/2): dWx = dWx
            elif(xk >= w):  # If we are at image width we must stop
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

            windows.append([ (wx_low,wy_high),(wx_high,wy_low) ])

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= wy_low) & (nonzeroy < wy_high) &
                         (nonzerox >= wx_low) &  (nonzerox < wx_high)).nonzero()[0]
            nPxls = len(good_inds)
            if verbose == True:
                print("\tCurrent Window [" + str(count) + "] Center: " + str(xk) + ", " + str(yk) + " ----- # Good Pixels = " + str(nPxls))

            # Record mean coordinates of pixels in window and update new window center
            if(nPxls >= threshold):
                xmean = np.int(np.mean(nonzerox[good_inds]))
                ymean = np.int(np.mean(nonzeroy[good_inds]))
                if mean_count == 0: mean_pxls.append(np.array([xmean,h]))
                else: mean_pxls.append(np.array([xmean,ymean]))

                dXmean = xmean - prev_xmean

                yk = yk - 2*dWy
                if dXmean < 0:
                    if dXmean < -3: xk = xk + abs(dXmean*2) + dWx/2
                    else: xk = xmean + abs(dXmean) + dWx/2
                else: xk = xmean + dWx/2

                if dXmean > 3: xk += 5
                elif dXmean == 3: xk += 3
                # Draw current window onto mask template
                my_low = ymean - dMy;          	my_high = ymean + dMy
                mx_low = xmean - dMx; 			mx_high = xmean + dMx
                masks.append([ (mx_low,my_high), (mx_high,my_low) ])
                # Update New window center coordinates
                # dchange =
                # tmpDx = dWx
                if dXmean > 3: dWx = dWxO + 6
                elif 1 < dXmean <= 3: dWx = dWxO + 3
                else: dWx = dWxO

                if(verbose): print("dXmean, dWx [%d]: %.3f, %d" % (mean_count,dXmean,dWx))
                # xk = xmean + dWx/2
                # yk = yk - 2*dWy

                # # Previous Method
                # xk = xmean + dWx
                # yk = ymean - 2*dWy
                mean_sum += dXmean
                mean_count += 1
                prev_xmean = xmean
                # dWx = dWxT
            else: flag_done = True
            count += 1
        if mean_count == 0: mean_count = 1
        avg_dxmean = mean_sum / float(mean_count)
        mean_pxls = np.array(mean_pxls)
        # lWidth = int(np.round(1.0*avg_dxmean))
        if avg_dxmean >= 1.5: lWidth = int(math.ceil(1.75*avg_dxmean))
        elif avg_dxmean == 1.0: lWidth = 2
        else: lWidth = int(math.ceil(1.0*avg_dxmean))
        # ==========================================================================
        if self.debug:
            print("Average Horizontal Change in [%d] detected ground pixels = %.2f (width=%d)" % (mean_pxls.shape[0],avg_dxmean,lWidth))
        if(mean_pxls.shape[0] >= min_ground_pixels and avg_dxmean > dxmean_thresh):
            if(draw_method == 1):
                pts = cv2.approxPolyDP(mean_pxls,3,0)
                cv2.polylines(black, [pts], 0, (255,255,255),lWidth)
                tmpPxl = np.array([ [mean_pxls[-1,0],mean_pxls[0,1]] ])
                tmpPxls = np.concatenate((mean_pxls,tmpPxl), axis=0)
                cv2.fillPoly(black, [tmpPxls], (255,255,255))
            else:
                for mask in masks:
                    cv2.rectangle(black,mask[0],mask[1],(255,255,255), cv2.FILLED)
            ground_detected = True
        else:
            ground_detected = False
        # ==========================================================================
        cv2.rectangle(black,(0,0),(dx,h),(255,255,255), cv2.FILLED)

        # ==========================================================================
        mask = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(mask)
        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[get_vmap_mask] --- Took %f seconds to complete" % (dt))

        return ground_detected, mask, mask_inv, mean_pxls, windows, dt
    def filter_first_umap_strip(self,umap_strip,max_value,thresholds,ratio_thresh=0.35):
        newStrips = []
        n = len(thresholds)
        strips = strip_image(umap_strip, nstrips=n)

        for i, strip in enumerate(strips):
            dratio = max_value/255.0
            tmpMax = np.max(strip)
            relRatio = tmpMax/float(max_value)
            if(dratio <= ratio_thresh):
                tmpGain = 10
                threshGain = 0.6
                idxPerc = (1+i)/float(n)
                if idxPerc > 0.5: threshGain = threshGain + 0.25
                # if i == 0: gainAdder = 0.35
                if i <= 1: gainAdder = (relRatio-dratio)
                else: gainAdder = 0
            else:
                tmpGain = 1
                threshGain = 1
                if i == 0: gainAdder = 0.05
                else: gainAdder = 0

            gain = thresholds[i]*threshGain*dratio*tmpGain + gainAdder
            tmpThresh = int(gain * max_value)
            # if self.debug: print("dratio, pre_gain, gain, thresh: %.3f, %.3f, %.3f, %d" % (dratio,threshGain,gain,tmpThresh))
            if self.debug: print("dratio, relRatio, pre_gain, gain, thresh: %.3f, %.3f, %.3f, %.3f, %d" % (dratio,relRatio, threshGain,gain,tmpThresh))
            print("Umap Strip [%d] Max, Thresholded: %d, %d" % (i,max_value,tmpThresh))
            # _,tmpStrip = cv2.threshold(strip, tmpThresh,255,cv2.THRESH_BINARY)
            _,tmpStrip = cv2.threshold(strip, tmpThresh,255,cv2.THRESH_TOZERO)

            newStrips.append(tmpStrip)

        newStrip = np.concatenate(newStrips, axis=0)
        return newStrip
    def test_filter_first_umap_strip(self,umap_strip,max_value,nSubStrips,ratio_thresh=0.25,base_thresh=35):
        try: umap_strip = cv2.cvtColor(umap_strip,cv2.COLOR_BGR2GRAY)
        except: print("[WARNING] test_filter_first_umap_strip() ------  Unnecessary Strip Color Conversion BGR -> Gray")
        n = nSubStrips
        strip = np.copy(umap_strip)
        hs, ws = strip.shape[:2];             dh = hs / n
        dead_strip = strip[0:dh, :]
        rest_strip = strip[dh:hs, :]
        oDead = np.copy(dead_strip); oRest = np.copy(rest_strip)

        hd, wd = dead_strip.shape[:2];        hr, wr = rest_strip.shape[:2]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(40,1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(75,3))
        tfa = int(0.75*max_value);
        # print("tfa: %d" %(tfa))

        tmpdead = dead_strip[0:3, :]
        tmprest = dead_strip[3:hd, :]

        # _, fa = cv2.threshold(tmpdead, tfa,255,cv2.THRESH_TOZERO) #THRESH_TOZERO

        tCLAH = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(1,20))
        dCl = tCLAH.apply(tmpdead)

        pc1 = dCl[0:1, :]
        pc2 = dCl[1:dCl.shape[0], :]

        # _,pc1 = cv2.threshold(pc1, 200,255,cv2.THRESH_TOZERO)
        _,pc1 = cv2.threshold(pc1, 256,255,cv2.THRESH_TOZERO)
        _,pc2 = cv2.threshold(pc2, 175,255,cv2.THRESH_TOZERO)

        # print("[DEBUG] test_filter_first_umap_strip() -----  Shapes: pc1 [%s], pc2 [%s]" % (str(pc1.shape),str(pc2.shape)))

        tmpdead = np.concatenate((pc1,pc2), axis=0)
        tmpdead = cv2.morphologyEx(tmpdead, cv2.MORPH_CLOSE, kernel)

        _,tmprest = cv2.threshold(tmprest, 55,255,cv2.THRESH_TOZERO)
        tmprest = cv2.morphologyEx(tmprest, cv2.MORPH_CLOSE, kernel2)

        dead_strip = np.concatenate((tmpdead,tmprest), axis=0);

        stripMax = np.max(strip)
        deadMax = np.max(dead_strip)
        restMax = np.max(rest_strip)

        relRatio = stripMax/float(max_value)
        dmRatio = deadMax / float(stripMax)
        rmRatio = restMax/float(stripMax)


        deadThreshGain = (stripMax - deadMax)/255.0
        deadThreshOffset = int(base_thresh*deadThreshGain)
        if not self.is_ground_present: deadThresh = deadThreshOffset
        else: deadThresh = base_thresh + deadThreshOffset

        dead_strip[dead_strip < deadThresh] = 0
        deadMax = np.max(dead_strip)
        # print("base, gain, offset, thresh: %d, %.2f, %d, %d" % (baseDeadThresh,deadThreshGain,deadThreshOffset,deadThresh))

        # Perform CLAHE
        claheDead = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(hd,wd))
        claheRest = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(hr/2,wr))

        # threshD = 7
        threshD = 50
        clD = claheDead.apply(dead_strip);                #copyCld = np.copy(clD)
        clR = claheRest.apply(rest_strip);                #copyClr = np.copy(clR)
        _,stripD = cv2.threshold(clD, threshD,255,cv2.THRESH_TOZERO)
        cldMax = np.max(stripD)
        clrMax = np.max(clR)

        if deadMax == 0: newDeadMax = 1
        else: newDeadMax = deadMax
        deadMaxGain = (cldMax - deadMax)/float(cldMax)
        restMaxGain = (clrMax - restMax)/float(clrMax)

        newStripMax = int(stripMax*restMaxGain)
        newRmRatio = (restMax)/float(newStripMax)

        gain1 = restMaxGain*ratio_thresh
        gain2 = gain1+ratio_thresh
        gain2 = gain2 + gain2*restMaxGain
        threshR = int(gain2*clrMax)
        threshR = threshR - int(gain1*threshR)
        _,stripR = cv2.threshold(clR, threshR,255,cv2.THRESH_TOZERO)

        newStrip = np.concatenate((stripD,stripR), axis=0)
        nStrip = np.copy(newStrip)
        # pStrip = np.copy(nStrip)
        ksize = (5,1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,ksize)
        pStrip = cv2.morphologyEx(newStrip, cv2.MORPH_CLOSE, kernel)
        disp = []

        try: newStrip = cv2.cvtColor(newStrip,cv2.COLOR_GRAY2BGR)
        except: print("[WARNING] test_filter_first_umap_strip() ------  Unnecessary Strip Color Conversion Gray -> BGR for \'newStrip\'")
        try: pStrip = cv2.cvtColor(pStrip,cv2.COLOR_GRAY2BGR)
        except: print("[WARNING] test_filter_first_umap_strip() ------  Unnecessary Strip Color Conversion Gray -> BGR for \'pStrip\'")

        return newStrip, strip, nStrip,pStrip, disp
    def test_filter_first_umap_strip2(self,umap_strip,max_value,nSubStrips):
        try: umap_strip = cv2.cvtColor(umap_strip,cv2.COLOR_BGR2GRAY)
        except: print("[WARNING] test_filter_first_umap_strip() ------  Unnecessary Strip Color Conversion BGR -> Gray")
        n = nSubStrips
        strip = np.copy(umap_strip)
        hs, ws = strip.shape[:2];             dh = hs / n

        deadzone = strip[0:dh, :];           oDead = np.copy(deadzone)
        rest_strip = strip[dh:hs, :];        oRest = np.copy(rest_strip)
        hd, wd = deadzone.shape[:2];         hr, wr = rest_strip.shape[:2]

        # ==========================================================================
        dzDeadStrip = deadzone[0:3, :]
        dzRest = deadzone[3:hd, :]

        subDeadZone = dzDeadStrip[0:1, :]
        subRestZone = dzDeadStrip[1:dzDeadStrip.shape[0], :]

        dzRestMax = np.max(dzRest);                 dzRestMean = np.mean(dzRest)
        mainDeadzoneMax = np.max(dzDeadStrip);      subRestZoneMax = np.max(subRestZone)
        mainDeadzoneMean = np.mean(dzDeadStrip);    subRestZoneMean = np.mean(subRestZone)

        ratioMainDead = mainDeadzoneMean/float(mainDeadzoneMax)
        ratioSubRest = subRestZoneMean/float(subRestZoneMax)
        tmpRatio = ratioSubRest + ratioMainDead*ratioSubRest

        thrSubDead = int((1-ratioMainDead)*mainDeadzoneMax)
        tmpThresh = int(tmpRatio*mainDeadzoneMax)

        _,subDeadZoneFiltered = cv2.threshold(subDeadZone, thrSubDead,255,cv2.THRESH_TOZERO)

        subRestZone1 = dzDeadStrip[1:2, :]
        subRestZone2 = dzDeadStrip[2:dzDeadStrip.shape[0], :]

        _,subRestZoneFiltered1 = cv2.threshold(subRestZone1, tmpThresh+10,255,cv2.THRESH_TOZERO)
        _,subRestZoneFiltered2 = cv2.threshold(subRestZone2, tmpThresh,255,cv2.THRESH_TOZERO)
        subRestZoneFiltered = np.concatenate((subRestZoneFiltered1,subRestZoneFiltered2), axis=0)


        subDzFiltered = np.concatenate((subDeadZoneFiltered,subRestZoneFiltered), axis=0);
        subDz = np.copy(subDzFiltered)

        subDzFilteredMax = np.max(subDzFiltered)
        subDzFilteredMean = np.mean(subDzFiltered)
        if subDzFilteredMax == 0: gain = 0
        else: gain = dzRestMax/float(subDzFilteredMax)

        ratioDzRest = dzRestMean/float(dzRestMax)
        ratioSubDzFiltered = subDzFilteredMean/float(subDzFilteredMax)
        tmpRatio = ratioDzRest + gain

        dzRestThresh = int((1-ratioDzRest)*dzRestMax)
        subThresh1 = int(tmpRatio*dzRestThresh)
        subThresh2 = int(tmpRatio*dzRestMax)
        combinedThresh = subThresh1 + subThresh2

        if not self.is_ground_present or 100 < combinedThresh < 255: tmpThresh = subThresh2
        elif combinedThresh >= 255: tmpThresh = int(0.5*255)
        else: tmpThresh = combinedThresh

        if subDzFilteredMax == 0: altMainThresh = 0
        else: altMainThresh = int(math.ceil(ratioSubDzFiltered*subDzFilteredMax))

        # if altMainThresh <= 2: altMainThresh = 7

        _,tmpDzRest = cv2.threshold(dzRest, tmpThresh,255,cv2.THRESH_TOZERO)

        subDzKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(30,1))
        subRestKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20,1))

        subDz = cv2.morphologyEx(subDz, cv2.MORPH_CLOSE, subDzKernel)
        subRest = cv2.morphologyEx(tmpDzRest, cv2.MORPH_CLOSE, subRestKernel)
        deadzone = np.concatenate((subDz,subRest), axis=0)

        tmpStrip = np.concatenate((deadzone,rest_strip), axis=0)
        tmpStripCopy = np.copy(tmpStrip)

        clahe = cv2.createCLAHE(clipLimit=10.0,tileGridSize=(hs,ws/4))
        stripCLAHE = clahe.apply(tmpStrip)

        tmpStripMax = np.max(tmpStrip);   tmpStripMean = np.mean(tmpStrip)
        claheMax = np.max(stripCLAHE);    claheMean = np.mean(stripCLAHE)
        meanGain = (claheMean - tmpStripMean)/float(tmpStripMean)

        ratioClahe = claheMean/float(claheMax)
        ratioTmpStrip = tmpStripMean/float(tmpStripMax)

        tmpRatio = ratioTmpStrip*meanGain
        tmpRatio1 = ratioClahe + tmpRatio

        claheThresh = int(ratioClahe*claheMax)
        subThresh = int(tmpRatio1*claheMax)
        tmpThresh = int(subThresh + tmpRatio1*claheThresh + math.ceil(ratioTmpStrip*claheMax))

        _,claheMask = cv2.threshold(stripCLAHE, tmpThresh,255,cv2.THRESH_BINARY)

        claheMaskKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(40,3))
        claheMask = cv2.morphologyEx(claheMask, cv2.MORPH_CLOSE, claheMaskKernel)

        resStrip = cv2.bitwise_and(tmpStripCopy,tmpStripCopy,mask=claheMask)
        resultant = resStrip[deadzone.shape[0]:resStrip.shape[0],:]

        mainStrip = np.concatenate((deadzone,resultant), axis=0)
        _,newStrip = cv2.threshold(mainStrip, altMainThresh,255,cv2.THRESH_TOZERO)

        try: newStrip = cv2.cvtColor(newStrip,cv2.COLOR_GRAY2BGR)
        except: print("[WARNING] test_filter_first_umap_strip() ------  Unnecessary Strip Color Conversion Gray -> BGR for \'newStrip\'")

        return newStrip, strip

    def second_umap_strip_filter(self, umap_strip, verbose=True):
        try: umap_strip = cv2.cvtColor(umap_strip,cv2.COLOR_BGR2GRAY)
        except: print("[WARNING] second_umap_strip_filter() ------  Unnecessary Strip Color Conversion BGR -> Gray")
        strip = np.copy(umap_strip)
        hs, ws = strip.shape[:2]
        # print("second_umap_strip_filter() --- Input Strip Shape: %s" % str(strip.shape[:2]))

        clahe = cv2.createCLAHE(clipLimit=10.0,tileGridSize=(hs,ws/4))
        claheStrip = clahe.apply(strip)

        mx0 = np.max(claheStrip);     mn0 = np.mean(claheStrip)
        mx1 = np.max(strip);   mn1 = np.mean(strip)
        maxGain = (mx1)/float(mx0)
        meanGain = (mn0 - mn1)/float(mn1)

        tmpRatio0 = mn0/float(mx0)
        tmpRatio1 = mn1/float(mx1)
        tmpRatio2 = tmpRatio0 + maxGain
        tmpRatio3 = tmpRatio1 + tmpRatio1*tmpRatio2

        tmpTH1 = int(tmpRatio2*mx0)
        tmpTH2 = int(math.ceil(tmpRatio3*mx1))
        testThresh = tmpTH1
        if tmpTH2 <= 1: restThresh = 3
        else: restThresh = tmpTH2

        if verbose:
            print("Means:\t\t%.2f, %.2f"% (mn0,mn1))
            print("Maxs:\t\t%d, %d"% (mx0,mx1))
            print("Mean Gain:\t%.3f" % meanGain)
            print("Max Gain:\t%.3f" % maxGain)
            plist("Tmp Ratios:\t",[tmpRatio0,tmpRatio1,tmpRatio2],dplace=3)
            plist("Tmp Threhsolds: ",[tmpTH1,tmpTH2])
            print("testThresh: %d" % testThresh)
            print("RestThresh: %d" % restThresh)

        _,strMask = cv2.threshold(claheStrip, testThresh,255,cv2.THRESH_BINARY)

        ks = (20,2)
        kernelM = cv2.getStructuringElement(cv2.MORPH_RECT,ks)
        strMask = cv2.morphologyEx(strMask, cv2.MORPH_CLOSE, kernelM)

        tmp = np.copy(strip)
        tmpStrip = cv2.bitwise_and(tmp,tmp,mask=strMask)
        newStrip = np.copy(tmpStrip)

        _,newStrip = cv2.threshold(newStrip, restThresh,255,cv2.THRESH_TOZERO)
        # print("second_umap_strip_filter() --- Output Strip Shape: %s" % str(newStrip.shape[:2]))

        try: newStrip = cv2.cvtColor(newStrip,cv2.COLOR_GRAY2BGR)
        except: print("[WARNING] second_umap_strip_filter() ------  Unnecessary Strip Color Conversion Gray -> BGR for \'newStrip\'")

        return newStrip

    def pipeline(self, _img, threshU1=7, threshU2=20,threshV2=70, timing=False):
        """
        ============================================================================
                                    Entire pipeline
        ============================================================================
        """
        dt = 0
        if(timing): t0 = time.time()
        # =========================================================================

        img = self.read_image(_img)

        kernelI = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelI)

        h, w = img.shape[:2]
        dead_x = self.dead_x; dead_y = self.dead_y

        raw_umap, raw_vmap, dt = self.get_uv_map(img)
        self.umap_raw = np.copy(raw_umap)
        self.vmap_raw = np.copy(raw_vmap)
        # =========================================================================
        deadzoneU = raw_umap[1:dead_y+1, :]
        _, deadzoneU = cv2.threshold(deadzoneU, 95, 255,cv2.THRESH_BINARY)

        cv2.rectangle(raw_umap,(0,0),(raw_umap.shape[1],dead_y),(0,0,0), cv2.FILLED)
        cv2.rectangle(raw_umap,(0,raw_umap.shape[0]-dead_y),(raw_umap.shape[1],raw_umap.shape[0]),(0,0,0), cv2.FILLED)

        cv2.rectangle(raw_vmap,(0,0),(dead_x, raw_vmap.shape[0]),(0,0,0), cv2.FILLED)
        cv2.rectangle(raw_vmap,(raw_vmap.shape[1]-dead_x,0),(raw_vmap.shape[1],raw_vmap.shape[0]),(0,0,0), cv2.FILLED)
        # =========================================================================
        if(self.flag_simulation): _, raw_umap = cv2.threshold(raw_umap, 35, 255,cv2.THRESH_TOZERO)
        self.umap_deadzoned = np.copy(raw_umap)
        self.vmap_deadzoned = np.copy(raw_vmap)
        _, raw_umap = cv2.threshold(raw_umap, 10, 255,cv2.THRESH_TOZERO)
        try:
            raw_umap = cv2.cvtColor(raw_umap,cv2.COLOR_GRAY2BGR)
            raw_vmap = cv2.cvtColor(raw_vmap,cv2.COLOR_GRAY2BGR)
        except:
            print("[WARNING] ------------  Unnecessary Raw Mappings Color Converting")
        # ==========================================================================
        #							U-MAP Specific Functions
        # ==========================================================================


        stripsU = strip_image(raw_umap, nstrips=6)
        self.stripsU_raw = np.copy(stripsU)

        _, stripU1 = cv2.threshold(stripsU[0], threshU1, 255,cv2.THRESH_BINARY)
        _, stripU2 = cv2.threshold(stripsU[1], threshU2, 255,cv2.THRESH_BINARY)
        _, stripU3 = cv2.threshold(stripsU[2], 30, 255,cv2.THRESH_BINARY)
        _, stripU4 = cv2.threshold(stripsU[3], 40, 255,cv2.THRESH_BINARY)
        _, stripU5 = cv2.threshold(stripsU[4], 40, 255,cv2.THRESH_BINARY)
        _, stripU6 = cv2.threshold(stripsU[5], 40, 255,cv2.THRESH_BINARY)

        # _, stripU1 = cv2.threshold(stripsU[0], threshU1, 255,cv2.THRESH_TOZERO)
        # _, stripU2 = cv2.threshold(stripsU[1], threshU2, 255,cv2.THRESH_TOZERO)
        # _, stripU3 = cv2.threshold(stripsU[2], 30, 255,cv2.THRESH_TOZERO)
        # _, stripU4 = cv2.threshold(stripsU[3], 40, 255,cv2.THRESH_TOZERO)
        # _, stripU5 = cv2.threshold(stripsU[4], 40, 255,cv2.THRESH_TOZERO)
        # _, stripU6 = cv2.threshold(stripsU[5], 40, 255,cv2.THRESH_TOZERO)

        hUs = stripU1.shape[0]
        blankStrip = np.zeros((hUs-dead_y,w),dtype=np.uint8)
        deadzone_mask = np.concatenate((deadzoneU,blankStrip), axis=0)
        try: deadzone_mask = cv2.cvtColor(deadzone_mask, cv2.COLOR_GRAY2BGR)
        except: print("[WARNING] ------------  Unnecessary Deadzone Image Color Converting")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
        stripU1 = cv2.morphologyEx(stripU1, cv2.MORPH_CLOSE, kernel)
        stripU2 = cv2.morphologyEx(stripU2, cv2.MORPH_CLOSE, kernel)
        stripU3 = cv2.morphologyEx(stripU3, cv2.MORPH_OPEN, kernel)

        kernelD = cv2.getStructuringElement(cv2.MORPH_RECT,(50,5))
        deadzone_mask = cv2.morphologyEx(deadzone_mask, cv2.MORPH_CLOSE, kernelD)
        stripU1 = cv2.addWeighted(stripU1, 1.0, deadzone_mask, 1.0, 0)

        self.stripsU_processed = np.copy([stripU1,stripU2,stripU3,stripU4,stripU5,stripU6])

        # umap = np.concatenate((stripU1,stripU2,stripU3,stripU4), axis=0)
        umap = np.concatenate((stripU1,stripU2,stripU3,stripU4,stripU5,stripU6), axis=0)
        try: umap = cv2.cvtColor(umap, cv2.COLOR_BGR2GRAY)
        except: print("[WARNING] ------------  Unnecessary Umap Color Converting")

        self.umap_processed = np.copy(umap)

        # ==========================================================================
        fCnts1,_ = self.find_contours(stripU1, 55.0, offset=(0,0))
        fCnts2,_ = self.find_contours(stripU2, 100.0, offset=(0,hUs))
        fCnts3,_ = self.find_contours(stripU3, 80.0, offset=(0,hUs*2))
        fCnts4,_ = self.find_contours(stripU4, 40.0, offset=(0,hUs*3))
        fCnts5,_ = self.find_contours(stripU5, 40.0, offset=(0,hUs*4))
        fCnts6,_ = self.find_contours(stripU6, 40.0, offset=(0,hUs*5))
        contours = fCnts1 + fCnts2 + fCnts3 + fCnts4 + fCnts5 + fCnts6
        xLims, dLims, _ = self.extract_contour_bounds(contours)

        self.xBounds = xLims
        self.disparityBounds = dLims
        self.filtered_contours = contours
        # ==========================================================================
        #							V-MAP Specific Functions
        # ==========================================================================
        stripsV = strip_image(raw_vmap, nstrips=5, horizontal_strips=False)
        self.stripsV_raw = np.copy(stripsV)

        _, stripV1 = cv2.threshold(stripsV[0], 5, 255,cv2.THRESH_BINARY)
        _, stripV2 = cv2.threshold(stripsV[1], threshV2, 255,cv2.THRESH_BINARY)
        _, stripV3 = cv2.threshold(stripsV[2], 40, 255,cv2.THRESH_BINARY)
        _, stripV4 = cv2.threshold(stripsV[3], 40, 255,cv2.THRESH_BINARY)
        _, stripV5 = cv2.threshold(stripsV[4], 40, 255,cv2.THRESH_BINARY)

        self.stripsV_processed = np.copy([stripV1,stripV2,stripV3,stripV4,stripV5])

        # newV = np.concatenate((stripV1,stripV2,stripV3,stripV4), axis=1)
        newV = np.concatenate((stripV1,stripV2,stripV3,stripV4,stripV5), axis=1)

        try: tmpV = cv2.cvtColor(newV, cv2.COLOR_BGR2GRAY)
        except:
            tmpV = newV
            print("[WARNING] ------------  Unnecessary Umap Color Converting")

        self.vmap_filtered = np.copy(tmpV)

        ground_detected, mask, maskInv,mPxls, ground_wins,_ = self.get_vmap_mask(newV, maxStep = 15)
        vmap = cv2.bitwise_and(newV,newV,mask=maskInv)

        try: vmap = cv2.cvtColor(vmap, cv2.COLOR_BGR2GRAY)
        except: print("[WARNING] ------------  Unnecessary Vmap Color Converting")

        self.vmask = np.copy(mask)
        self.vmap_processed = np.copy(vmap)
        # =========================================================================

        # print("[INFO] Beginning Obstacle Search....")
        obs, obsU, ybounds, dbounds, windows, nObs = self.find_obstacles(vmap, dLims, xLims, ground_detected=ground_detected)

        # =========================================================================
        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[uv_pipeline] --- Took %f seconds to complete" % (dt))

        self.nObs = nObs
        self.obstacles = obs
        self.obstacles_umap = obsU
        self.yBounds = ybounds
        self.dbounds = dbounds
        self.ground_pxls = mPxls
        self.windows_obstacles = windows
        self.windows_ground = ground_wins
        return 0

    def pipelineTest(self, _img, threshU1=0.25, threshU2=0.3, threshV1=5, threshV2=70, timing=False):
        """
        ============================================================================
                                Entire Test pipeline
        ============================================================================
        """
        dt = 0
        threshsU = [threshU1, threshU2, 0.3, 0.7,0.5,0.5]
        threshsV = [threshV1, threshV2, 40,60,60]
        threshsCnt = [15.0,100.0,80.0,80.0,40.0,40.0]


        if(timing): t0 = time.time()
        # =========================================================================

        img = self.read_image(_img)

        nThreshsU = int(math.ceil((self.dmax/256.0) * len(threshsU)))
        nThreshsV = int(math.ceil((self.dmax/256.0) * len(threshsV)))

        kernelI = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelI)

        h, w = img.shape[:2]
        dead_x = self.dead_x; dead_y = self.dead_y

        raw_umap, raw_vmap, dt = self.get_uv_map(img)
        # print("[INFO] pipelineTest() ---- generated vmap shape (H,W,C): %s" % (str(raw_vmap.shape)))
        self.umap_raw = np.copy(raw_umap)
        self.vmap_raw = np.copy(raw_vmap)
        # =========================================================================
        cv2.rectangle(raw_umap,(0,raw_umap.shape[0]-dead_y),(raw_umap.shape[1],raw_umap.shape[0]),(0,0,0), cv2.FILLED)
        cv2.rectangle(raw_vmap,(raw_vmap.shape[1]-dead_x,0),(raw_vmap.shape[1],raw_vmap.shape[0]),(0,0,0), cv2.FILLED)
        # =========================================================================
        # if(self.flag_simulation): _, raw_umap = cv2.threshold(raw_umap, 35, 255,cv2.THRESH_TOZERO)
        self.umap_deadzoned = np.copy(raw_umap)
        self.vmap_deadzoned = np.copy(raw_vmap)
        try:
            raw_umap = cv2.cvtColor(raw_umap,cv2.COLOR_GRAY2BGR)
            raw_vmap = cv2.cvtColor(raw_vmap,cv2.COLOR_GRAY2BGR)
        except: print("[WARNING] ------------  Unnecessary Raw Mappings Color Converting")

        # ==========================================================================
        #							V-MAP Specific Functions
        # ==========================================================================
        ground_detected, mask, maskInv,mPxls, ground_wins,_ = self.get_vmap_mask(raw_vmap, maxStep = self.testMaxGndStep, window_size=[15,15])

        stripsPv = []
        stripsV = strip_image(raw_vmap, nstrips=nThreshsV, horizontal_strips=False)
        self.stripsV_raw = list(stripsV)
        # print("[INFO] raw_vmap shape: %s" % (str(raw_vmap.shape)))

        for i, strip in enumerate(stripsV):
            _, tmpStrip = cv2.threshold(strip, threshsV[i], 255, cv2.THRESH_TOZERO)
            stripsPv.append(tmpStrip)

        vsThs = np.array([
            [0.25, 0.0],
            [0.5, 0.0]
        ])

        tmpMax = np.max(stripsPv[0])
        stripsPv[0][stripsPv[0] < np.max(stripsPv[0]*0.035)] = 0

        top_strip = stripsPv[0][0:100, :]
        bot_strip = stripsPv[0][100:stripsPv[0].shape[0], :]

        top_strip[top_strip < tmpMax*vsThs[0,0]] = 0
        bot_strip[bot_strip < tmpMax*vsThs[0,1]] = 0
        stripsPv[0] = np.concatenate((top_strip,bot_strip), axis=0)

        dH = self.dH;        dHplus = self.dHplus;      dths = self.dThs
        dw = self.dW;        vdTh = [dths[3], 0]

        dead_strip = stripsPv[0][:, 0:dw]
        rest_strip = stripsPv[0][:,dw:stripsPv[0].shape[1]]

        # print("[INFO] Shapes stripsPv[0], dead_strip, rest_strip: %s, %s, %s" % (str(stripsPv[0].shape), str(dead_strip.shape),str(rest_strip.shape)))

        dead_strip[dead_strip < tmpMax*vdTh[0]] = 0
        rest_strip[rest_strip < tmpMax*vdTh[1]] = 0
        stripsPv[0] = np.concatenate((dead_strip,rest_strip), axis=1)

        # print("[INFO] stripsPv[0] shape: %s" % (str(stripsPv[0].shape)))

        # stripsPv[1][stripsPv[1] < np.max(stripsPv[1]*0.9)] = 0
        try:
            tmpMax = np.max(stripsPv[1])
            tH = stripsPv[1].shape[0]
            top_strip = stripsPv[1][0:tH/3, :]
            bot_strip = stripsPv[1][tH/3:tH, :]

            top_strip[top_strip < tmpMax*vsThs[1,0]] = 0
            bot_strip[bot_strip < tmpMax*vsThs[1,1]] = 0
            stripsPv[1] = np.concatenate((top_strip,bot_strip), axis=0)
        except: pass

        self.stripsV_processed = list(stripsPv)
        newV = np.concatenate(stripsPv, axis=1)
        # print("[INFO] newV shape: %s" % (str(newV.shape)))

        try: tmpV = cv2.cvtColor(newV, cv2.COLOR_BGR2GRAY)
        except:
            tmpV = newV
            print("[WARNING] ------------  Unnecessary Umap Color Converting")

        self.vmap_filtered = np.copy(tmpV)

        # ground_detected, mask, maskInv,mPxls, ground_wins,_ = self.get_vmap_mask(newV, maxStep = self.testMaxGndStep, window_size=[10,10])
        tmp = np.copy(tmpV)
        # print(tmp.shape, maskInv.shape)
        vmap = cv2.bitwise_and(tmp,tmp,mask=maskInv)

        # try: vmap = cv2.cvtColor(vmap, cv2.COLOR_BGR2GRAY)
        # except: print("[WARNING] ------------  Unnecessary Vmap Color Converting")

        self.vmask = np.copy(mask)
        self.vmap_processed = np.copy(vmap)
        self.is_ground_present = ground_detected
        # =========================================================================

        # ==========================================================================
        #							U-MAP Specific Functions
        # ==========================================================================

        stripsU = strip_image(raw_umap, nstrips=nThreshsU)
        self.stripsU_raw = list(stripsU)

        stripsPu = []
        for i, strip in enumerate(stripsU):
            if i == 0:
                maxdisparity = np.max(raw_umap)
                stripThreshs = [0.15, 0.25, 0.075, 0.065,0.05,0.025]
                # tmpStrip = self.filter_first_umap_strip(strip,maxdisparity,stripThreshs)
                nSubStrips = int(math.ceil((self.dmax/256.0) * len(stripThreshs)))
                # _,_,_,tmpStrip,_ = self.test_filter_first_umap_strip(strip,maxdisparity,nSubStrips, ratio_thresh=self.testRestThreshRatio)
                tmpStrip,_ = self.test_filter_first_umap_strip2(strip,maxdisparity,nSubStrips)
                # print("tmpStrip Shape: %s" % str(tmpStrip.shape))
            elif i == 1:
                tmpStrip = self.second_umap_strip_filter(strip)
                # print("tmpStrip Shape: %s" % str(tmpStrip.shape))
            else:
                tmpMax = np.max(strip)
                tmpThresh = threshsU[i] * tmpMax
                # if(self.debug): print("Umap Strip [%d] Max, Thresholded: %d, %d" % (i,tmpMax,tmpThresh))
                _, tmpStrip = cv2.threshold(strip, tmpThresh, 255,cv2.THRESH_TOZERO)
                # _, tmpStrip = cv2.threshold(strip, threshsU[i], 255,cv2.THRESH_BINARY)
                # print("tmpStrip Shape: %s" % str(tmpStrip.shape))
            stripsPu.append(tmpStrip)

        ksz1 = self.kszs[0]
        kernel0 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,2))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,ksz1)
        stripsPu[0] = cv2.morphologyEx(stripsPu[0], cv2.MORPH_CLOSE, kernel0)
        stripsPu[1] = cv2.morphologyEx(stripsPu[1], cv2.MORPH_CLOSE, kernel)
        if len(stripsPu)>=3: stripsPu[2] = cv2.morphologyEx(stripsPu[2], cv2.MORPH_OPEN, kernel)

        hUs,w = stripsPu[0].shape[:2]

        self.stripsU_processed = list(stripsPu)
        umap = np.concatenate(stripsPu, axis=0)

        try: umap = cv2.cvtColor(umap, cv2.COLOR_BGR2GRAY)
        except: print("[WARNING] ------------  Unnecessary Umap Color Converting")
        self.umap_processed = np.copy(umap)
        # ==========================================================================

        # contours = []
        # for i, strip in enumerate(stripsPu):
        #     contours,_ += self.find_contours(strip, threshsCnt[i], offset=(0,hUs*i),debug=self.debug)
        if ground_detected:
            contour_thresh = self.testHighCntThresh
            if(self.debug): print("[INFO] pipelineTest ---- Ground Detected -> filtering contours w/ [%.2f] threshhold" % (contour_thresh))
        else:
            contour_thresh = self.testLowCntThresh
            if(self.debug): print("[INFO] pipelineTest ---- Ground Not Detected -> filtering contours w/ [%.2f] threshhold" % (contour_thresh))

        contours,_ = self.find_contours(umap, contour_thresh,debug=self.debug_contours)

        self.contours = contours
        xLims, dLims, _ = self.extract_contour_bounds(contours)
        self.xBounds = xLims
        self.disparityBounds = dLims
        self.filtered_contours = contours

        # # ==========================================================================
        # #							V-MAP Specific Functions
        # # ==========================================================================


        # print("[INFO] Beginning Obstacle Search....")
        obs, obsU, ybounds, dbounds, windows, nObs = self.find_obstacles(vmap, dLims, xLims, ground_detected=ground_detected,verbose=self.debug_obstacle_search)

        # =========================================================================
        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[uv_pipeline] --- Took %f seconds to complete" % (dt))

        self.nObs = nObs
        self.obstacles = obs
        self.obstacles_umap = obsU
        self.yBounds = ybounds
        self.dbounds = dbounds
        self.ground_pxls = mPxls
        self.windows_obstacles = windows
        self.windows_ground = ground_wins
        return 0


    def umap_displays(self, border_color=(255,0,255)):
        """
        ============================================================================
        	                      PLACEHOLDER
        ============================================================================
        """
        borderb = np.ones((1,self.w,3),dtype=np.uint8); borderb[:] = border_color
        borders = np.ones((self.h,1,3),dtype=np.uint8); borders[:] = border_color
        borderb2 = np.ones((1,self.w,3),dtype=np.uint8); borderb2[:] = (255,255,255)

        tmp = []
        for s in self.stripsU_raw:
            s = cv2.applyColorMap(s,cv2.COLORMAP_PARULA)
            tmp.append(s)
            tmp.append(borderb)
        dispU1 = np.concatenate(tmp, axis=0)

        tmp = []
        for s in self.stripsU_processed:
            s = cv2.applyColorMap(s,cv2.COLORMAP_PARULA)
            tmp.append(s)
            tmp.append(borderb)
        dispU2 = np.concatenate(tmp, axis=0)

        for cnt in self.filtered_contours:
            cv2.drawContours(dispU1, [cnt], 0, (255,255,255), 1)
            cv2.drawContours(dispU2, [cnt], 0, (255,255,255), 1)

        comp = np.concatenate((dispU1,borderb2,dispU2), axis=0)

        return dispU1, dispU2, comp
    def vmap_displays(self, border_color=(255,0,255)):
        borders = np.ones((self.h,1,3),dtype=np.uint8); borders[:] = border_color
        borders2 = np.ones((self.h,10,3),dtype=np.uint8); borders2[:] = (255,255,255)

        tmp = []
        for s in self.stripsV_raw:
            s = cv2.applyColorMap(s,cv2.COLORMAP_PARULA)
            tmp.append(s)
            tmp.append(borders)
        dispV1 = np.concatenate(tmp, axis=1)

        tmp = []
        for s in self.stripsV_processed:
            s = cv2.applyColorMap(s,cv2.COLORMAP_PARULA)
            tmp.append(s)
            tmp.append(borders)
        dispV2 = np.concatenate(tmp, axis=1)

        for pxl in self.ground_pxls:
            cv2.circle(dispV1,(pxl[0],pxl[1]),2,(255,0,255),-1)
            cv2.circle(dispV2,(pxl[0],pxl[1]),2,(255,0,255),-1)

        borderb = np.ones((10,dispV2.shape[1],3),dtype=np.uint8); borderb[:] = (255,255,255)

        for wins in self.windows_obstacles:
            for win in wins:
                cv2.rectangle(dispV1,win[0],win[1],(255,255,0), 1)
                cv2.rectangle(dispV2,win[0],win[1],(255,255,0), 1)

        # disp_ground = cv2.cvtColor(np.copy(vmap), cv2.COLOR_GRAY2BGR)
        # for win in ground_wins:
        #     cv2.rectangle(disp_ground,win[0],win[1],(255,255,0), 1)

        comp = np.concatenate((dispV1,borders2,dispV2), axis=1)
        return dispV1, dispV2, comp
    def calculate_distance(self, umap, xs, ds, ys, focal=[462.138,462.138],
        baseline=0.055, dscale=0.001, pp=[320.551,232.202], dsbuffer=1,
        use_principle_point=True, use_disparity_buffer=True, verbose=False):

        nonzero = umap.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if(use_disparity_buffer): ds[0] = ds[0] - dsbuffer

        good_inds = ((nonzeroy >= ds[0]) & (nonzeroy < ds[1]) &
                     (nonzerox >= xs[0]) &  (nonzerox < xs[1])).nonzero()[0]

        xmean = np.int(np.mean(nonzerox[good_inds]))
        dmean = np.mean(nonzeroy[good_inds])
        ymean = np.mean(ys)
        z = dmean*(65535/255)*dscale

        if use_principle_point: px, py = pp[:2]
        else: px = py = 0

        x = ((xmean - px)/focal[0])*z
        y = ((ymean - py)/focal[1])*z

        dist = np.sqrt(x*x+z*z)

        if(verbose): print("X, Y, Z: %.3f, %.3f, %.3f" % (x,y, dist))

        return dist,x,y,z
    def calculate_rotation_matrix(self,eulers):
        r = eulers[0]; p = eulers[1]; y = eulers[2];
        R11 = np.cos(p)*np.cos(y)
        R12 = np.cos(p)*np.sin(y)
        R13 = -np.sin(p)
        R1 = [R11, R12, R13]

        R21 = -(np.cos(r)*np.sin(y)) + (np.sin(r)*np.sin(p)*np.cos(y))
        R22 = (np.cos(r)*np.cos(y)) + np.sin(r)*np.sin(p)*np.sin(y)
        R23 = np.sin(r)*np.cos(p)
        R2 = [R21,R22,R23]

        R31 = (np.sin(r)*np.sin(y))+(np.cos(r)*np.sin(p)*np.cos(y))
        R32 = -(np.sin(r)*np.cos(y)) + (np.cos(r)*np.sin(p)*np.sin(y))
        R33 = np.cos(r)*np.cos(p)
        R3 = [R31,R32,R33]

        rotation = np.array([R1,R2,R3])
        return rotation
    def transform_pixel_to_world(self, rotation,pixel,translation, verbose=False):
        Rinv = np.linalg.inv(rotation)
        pos = np.dot(Rinv,pixel)
        position = pos - translation
        if(verbose): print("Projected Position (X,Y,Z): %s" % (', '.join(map(str, np.around(position,3)))) )
        return position

    def read_image(self,_img):
        """
        ============================================================================
            Load a depth image from the specified path only if the given path
            is different from the previously used path
        ============================================================================
        """
        if(type(_img) is np.ndarray): img = _img
        else: img = cv2.imread(_img,cv2.IMREAD_GRAYSCALE)

        # Get stats on original image
        self.h, self.w = img.shape[:2]
        self.dmax = np.max(img) + 1
        self.img = np.copy(img)
        return img
