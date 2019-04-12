
# cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L)
# cap2 = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L)
#===============================
import logging
logging.basicConfig(level=logging.INFO)

import time
import numpy as np
import cv2, Image
import pyrealsense as pyrs
from pyrealsense.constants import rs_option


with pyrs.Service() as serv:
    with serv.Device(depth_control_preset=1) as dev:


        try:  # set custom gain/exposure values to obtain good depth image
            custom_options = [(rs_option.RS_OPTION_R200_LR_EXPOSURE, 100.0),
                              (rs_option.RS_OPTION_R200_LR_GAIN, 137.0)]
            dev.set_device_options(*zip(*custom_options))
        except pyrs.RealsenseError:
            pass  # options are not available on all devices

        dev.apply_ivcam_preset(0)
        cnt = 0
        last = time.time()
        smoothing = 0.9
        fps_smooth = 30
        intr = dev.depth_intrinsics
        Fx = intr.fx
        extr = dev.get_device_extrinsics(2,3)
        baseline = extr.translation[0]
        print(Fx,baseline,1/dev.depth_scale)
        while True:

            cnt += 1
            if (cnt % 10) == 0:
                now = time.time()
                dt = now - last
                fps = 10/dt
                fps_smooth = (fps_smooth * smoothing) + (fps * (1.0-smoothing))
                last = now

            dev.wait_for_frames()
            c = dev.color
            cc = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)

            h,w,ch = cc.shape
            dumpImg = np.zeros((h, w, ch))

            d_raw = dev.depth * dev.depth_scale# * 1000
            # Depth scale = 1.0000000475
            # print(d_raw[240,:])
            d_raw = d_raw.astype(np.uint8)
            norm_image = cv2.normalize(d_raw, dumpImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            norm_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB)
            depth_raw = cv2.applyColorMap(d_raw, cv2.COLORMAP_RAINBOW)

            # kernel = np.ones((5,5),np.float32)/25
            # depth = cv2.filter2D(d_raw,-1,kernel)
            # depth = cv2.GaussianBlur(d_raw,(20,20),0)

            depth = cv2.medianBlur(d_raw, 5)
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_RAINBOW)
            blank = np.zeros(cc.shape, np.uint8)

            cd1 = np.concatenate((c, norm_image), axis=1)
            cd2 = np.concatenate((depth_raw,depth), axis=1)
            cd = np.concatenate((cd1, cd2), axis=0)

            cv2.putText(cd, str(fps_smooth)[:4], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

            # cv2.namedWindow('image',cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('depth',cv2.WINDOW_NORMAL)
            cv2.namedWindow('image',cv2.WINDOW_NORMAL)
            # cv2.imshow('depth', dev.depth)
            cv2.imshow('depth', cc)
            cv2.imshow('image', cd)
            # cv2.imshow('Depths', cmpD)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
