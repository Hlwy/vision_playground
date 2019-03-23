import numpy as np
import os, sys, cv2, time
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join('..', '')))

from utils.seg_utils import *

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
        self.vmap_search_window_height = 20
        self.mask_size = [20,40]
        self.dead_x = 2
        self.dead_y = 10
        self.dH = 14
        self.dW = 0
        self.dThs = [0.4, 0.1, 0.15]
        # Counters
        self.nObs = 0
        # Flags
        self.flag_simulation = False
        self.debug = True
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
    def find_contours(self, _umap, threshold = 30.0, threshold_method = "perimeter", offset=(0,0), debug=False):
        """
        ============================================================================
        	Find contours in image and filter out those above a certain threshold
        ============================================================================
        """
        try: umap = cv2.cvtColor(_umap,cv2.COLOR_BGR2GRAY)
        except:
            umap = _umap
            print("[WARNING] find_contours --- Unnecessary Image Color Converting")

        _, contours, hierarchy = cv2.findContours(umap,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE,offset=offset)

        if(threshold_method == "perimeter"):
            filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt,True) >= threshold]
            if(debug):
                raw_perimeters = [cv2.arcLength(cnt,True) for cnt in contours]
                filtered_perimeters = [cv2.arcLength(cnt,True) for cnt in filtered_contours]
                print("Raw Contour Perimeters:",raw_perimeters)
                print("Filtered Contour Perimeters:",filtered_perimeters)
        elif(threshold_method == "area"):
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= threshold]
            if(debug):
                raw_areas = [cv2.contourArea(cnt) for cnt in contours]
                filtered_areas = [cv2.contourArea(cnt) for cnt in filtered_contours]
                print("Raw Contour Areas:",raw_areas)
                print("Filtered Contour Areas:",filtered_areas)
        else:
            print("[ERROR] find_contours --- Unsupported filtering method!")

        return filtered_contours
    def find_obstacles(self, vmap, dLims, xLims, search_thresholds = (3,30), verbose=False):
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
            if self.debug: print("Y Limits: ", ys)
            if(len(ys) <= 1):
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
            if(dWx <= 1): dWx = 2
        else: dWx, dWy = np.int32(window_size)/2

        if(yk <= 0): yk = 0 + dWy
        if(verbose): print("Starting Location: (%d, %d)" % (xk, yk) )

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

        print("[%d] Good Windows" % nWindows)
        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[obstacle_search] --- Took %f seconds to complete" % (dt))

        return yLims, windows, dt
    def get_vmap_mask(self, vmap, threshold=20, min_ground_pixels=5, dxmean_thresh=1.0, maxStep=14, deltas=(0,20), mask_size = [10,30], window_size = [10,30], draw_method=1, verbose=False, timing=False):
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
        dMy,dMx = np.int32(mask_size)/2
        dx,dy = np.int32(deltas)
        # ==========================================================================

        # Create a black template to create mask with
        black = np.zeros((h,w,3),dtype=np.uint8)

        if(timing): t0 = time.time()
        # ==========================================================================

        # Take the bottom strip of the input image to find potential points to start sliding from
        y0 = abs(int(h-dy))
        hist = np.sum(vmap[y0:h,0:w], axis=0)
        x0 = abs(int(np.argmax(hist[:,0])))
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
                mean_pxls.append(np.array([xmean,ymean]))

                # Draw current window onto mask template
                my_low = ymean - dMy;          	my_high = ymean + dMy
                mx_low = xmean - dMx; 			mx_high = xmean + dMx
                masks.append([ (mx_low,my_high), (mx_high,my_low) ])
                # Update New window center coordinates
                xk = xmean + dWx
                yk = ymean - 2*dWy
                dXmean = xmean - prev_xmean
                mean_sum += dXmean
                mean_count += 1
                prev_xmean = xmean
            else: flag_done = True
            count += 1
        avg_dxmean = mean_sum / float(mean_count)
        mean_pxls = np.array(mean_pxls)
        # ==========================================================================
        if self.debug:
            print("Average Horizontal Change in [%d] detected ground pixels = %.2f" % (mean_pxls.shape[0],avg_dxmean))
        if(mean_pxls.shape[0] >= min_ground_pixels and avg_dxmean > dxmean_thresh):
            if(draw_method == 1):
                pts = cv2.approxPolyDP(mean_pxls,3,0)
                cv2.polylines(black, [pts], 0, (255,255,255),int(1.0*avg_dxmean))
            else:
                for mask in masks:
                    cv2.rectangle(black,mask[0],mask[1],(255,255,255), cv2.FILLED)
        # ==========================================================================
        cv2.rectangle(black,(0,0),(dx,h),(255,255,255), cv2.FILLED)

        # ==========================================================================
        mask = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(mask)
        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[get_vmap_mask] --- Took %f seconds to complete" % (dt))

        return mask, mask_inv, mean_pxls, windows, dt

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
        fCnts1 = self.find_contours(stripU1, 55.0, offset=(0,0))
        fCnts2 = self.find_contours(stripU2, 100.0, offset=(0,hUs))
        fCnts3 = self.find_contours(stripU3, 80.0, offset=(0,hUs*2))
        fCnts4 = self.find_contours(stripU4, 40.0, offset=(0,hUs*3))
        fCnts5 = self.find_contours(stripU5, 40.0, offset=(0,hUs*4))
        fCnts6 = self.find_contours(stripU6, 40.0, offset=(0,hUs*5))
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

        mask, maskInv,mPxls, ground_wins,_ = self.get_vmap_mask(newV, maxStep = 15)
        vmap = cv2.bitwise_and(newV,newV,mask=maskInv)

        try: vmap = cv2.cvtColor(vmap, cv2.COLOR_BGR2GRAY)
        except: print("[WARNING] ------------  Unnecessary Vmap Color Converting")

        self.vmask = np.copy(mask)
        self.vmap_processed = np.copy(vmap)
        # =========================================================================

        # print("[INFO] Beginning Obstacle Search....")
        obs, obsU, ybounds, dbounds, windows, nObs = self.find_obstacles(vmap, dLims, xLims)

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

    def pipelineTest(self, _img, threshU1=11, threshU2=6, threshV1=5, threshV2=70, timing=False):
        """
        ============================================================================
                                Entire Test pipeline
        ============================================================================
        """
        dt = 0
        threshsU = [threshU1, threshU2, 30, 40,40,40]
        threshsV = [threshV1, threshV2, 40,40,40]
        threshsCnt = [55.0,100.0,80.0,40.0,40.0,40.0]

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
        # _, raw_umap = cv2.threshold(raw_umap, 10, 255,cv2.THRESH_TOZERO)
        try:
            raw_umap = cv2.cvtColor(raw_umap,cv2.COLOR_GRAY2BGR)
            raw_vmap = cv2.cvtColor(raw_vmap,cv2.COLOR_GRAY2BGR)
        except:
            print("[WARNING] ------------  Unnecessary Raw Mappings Color Converting")
        # ==========================================================================
        #							U-MAP Specific Functions
        # ==========================================================================

        stripsU = strip_image(raw_umap, nstrips=len(threshsU))
        self.stripsU_raw = np.copy(stripsU)

        stripsPu = []
        for i, strip in enumerate(stripsU):
            # _, tmpStrip = cv2.threshold(strip, threshsU[i], 255,cv2.THRESH_BINARY)
            _, tmpStrip = cv2.threshold(strip, threshsU[i], 255,cv2.THRESH_TOZERO)
            stripsPu.append(tmpStrip)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
        stripsPu[0] = cv2.morphologyEx(stripsPu[0], cv2.MORPH_CLOSE, kernel)
        stripsPu[1] = cv2.morphologyEx(stripsPu[1], cv2.MORPH_CLOSE, kernel)
        stripsPu[2] = cv2.morphologyEx(stripsPu[2], cv2.MORPH_OPEN, kernel)

        hUs,w = stripsPu[0].shape[:2]
        blankStrip = np.zeros((hUs-dead_y,w),dtype=np.uint8)
        deadzone_mask = np.concatenate((deadzoneU,blankStrip), axis=0)
        try: deadzone_mask = cv2.cvtColor(deadzone_mask, cv2.COLOR_GRAY2BGR)
        except: print("[WARNING] ------------  Unnecessary Deadzone Image Color Converting")

        kernelD = cv2.getStructuringElement(cv2.MORPH_RECT,(50,5))
        deadzone_mask = cv2.morphologyEx(deadzone_mask, cv2.MORPH_CLOSE, kernelD)
        stripsPu[0] = cv2.addWeighted(stripsPu[0], 1.0, deadzone_mask, 1.0, 0)


        dH = self.dH;    dths = self.dThs
        tmpMax = np.max(stripsPu[0])
        dead_strip = stripsPu[0][0:dH, :]
        rest_strip = stripsPu[0][dH:stripsPu[0].shape[0], :]

        usThs = np.array([ [dths[0], dths[1]] ])

        dead_strip[dead_strip < tmpMax*usThs[0,0]] = 0
        rest_strip[rest_strip < tmpMax*usThs[0,1]] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(75,1))
        rest_strip = cv2.morphologyEx(rest_strip, cv2.MORPH_CLOSE, kernel)
        stripsPu[0] = np.concatenate((dead_strip,rest_strip), axis=0)


        self.stripsU_processed = np.copy(stripsPu)
        umap = np.concatenate(stripsPu, axis=0)
        try: umap = cv2.cvtColor(umap, cv2.COLOR_BGR2GRAY)
        except: print("[WARNING] ------------  Unnecessary Umap Color Converting")
        self.umap_processed = np.copy(umap)
        # ==========================================================================

        contours = []
        for i, strip in enumerate(stripsPu):
            contours += self.find_contours(strip, threshsCnt[i], offset=(0,hUs*i))

        xLims, dLims, _ = self.extract_contour_bounds(contours)
        self.xBounds = xLims
        self.disparityBounds = dLims
        self.filtered_contours = contours

        # ==========================================================================
        #							V-MAP Specific Functions
        # ==========================================================================

        stripsV = strip_image(raw_vmap, nstrips=len(threshsV), horizontal_strips=False)
        self.stripsV_raw = np.copy(stripsV)

        stripsPv = []
        for i, strip in enumerate(stripsV):
            # _, tmpStrip = cv2.threshold(strip, threshsV[i], 255,cv2.THRESH_BINARY)
            _, tmpStrip = cv2.threshold(strip, threshsV[i], 255,cv2.THRESH_TOZERO)
            stripsPv.append(tmpStrip)

        vsThs = np.array([
            [0.1, 0.0],
            [0.8, 0.0]
        ])

        tmpMax = np.max(stripsPv[0])
        stripsPv[0][stripsPv[0] < np.max(stripsPv[0]*0.035)] = 0

        top_strip = stripsPv[0][0:100, :]
        bot_strip = stripsPv[0][100:stripsPv[0].shape[0], :]

        top_strip[top_strip < tmpMax*vsThs[0,0]] = 0
        bot_strip[bot_strip < tmpMax*vsThs[0,1]] = 0
        stripsPv[0] = np.concatenate((top_strip,bot_strip), axis=0)

        dw = self.dW;        vdTh = [dths[2], 0]

        dead_strip = stripsPv[0][:, 0:dw]
        rest_strip = stripsPv[0][:,dw:stripsPv[0].shape[1]]

        dead_strip[dead_strip < tmpMax*vdTh[0]] = 0
        rest_strip[rest_strip < tmpMax*vdTh[1]] = 0
        stripsPv[0] = np.concatenate((dead_strip,rest_strip), axis=1)

        # stripsPv[1][stripsPv[1] < np.max(stripsPv[1]*0.9)] = 0
        tmpMax = np.max(stripsPv[1])
        tH = stripsPv[1].shape[0]
        top_strip = stripsPv[1][0:tH/3, :]
        bot_strip = stripsPv[1][tH/3:tH, :]

        top_strip[top_strip < tmpMax*vsThs[1,0]] = 0
        bot_strip[bot_strip < tmpMax*vsThs[1,1]] = 0
        stripsPv[1] = np.concatenate((top_strip,bot_strip), axis=0)

        self.stripsV_processed = np.copy(stripsPv)
        newV = np.concatenate(stripsPv, axis=1)
        try: tmpV = cv2.cvtColor(newV, cv2.COLOR_BGR2GRAY)
        except:
            tmpV = newV
            print("[WARNING] ------------  Unnecessary Umap Color Converting")

        self.vmap_filtered = np.copy(tmpV)

        mask, maskInv,mPxls, ground_wins,_ = self.get_vmap_mask(newV, maxStep = 16)
        vmap = cv2.bitwise_and(newV,newV,mask=maskInv)

        try: vmap = cv2.cvtColor(vmap, cv2.COLOR_BGR2GRAY)
        except: print("[WARNING] ------------  Unnecessary Vmap Color Converting")

        self.vmask = np.copy(mask)
        self.vmap_processed = np.copy(vmap)
        # =========================================================================

        # print("[INFO] Beginning Obstacle Search....")
        obs, obsU, ybounds, dbounds, windows, nObs = self.find_obstacles(vmap, dLims, xLims,verbose=False)

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
