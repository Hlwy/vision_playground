import numpy as np
import cv2, time, pprint
import pyrealsense as pyrs
from pyrealsense.constants import rs_option
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)

class CameraR200(object):
    def __init__(self):
        self.flag_save = 0
        self.rgb_img = None
        self.depth_img = None
        self.count = 0
        self.last = time.time()
        self.smoothing = 0.9
        self.fps_smooth = 60

        # Attempt to establish camera service
        try: self.srv = pyrs.Service()
        except:
            self.srv = None
            print("[ERROR] Could not establish CameraR200 Service!")

        # Attempt to connect to camera device
        try: self.dev = self.srv.Device(depth_control_preset=1)
        except:
            self.dev = None
            print("[ERROR] Could not establish CameraR200 Device!")
        # Attempt to configure camera settings
        self.apply_configuration()

        if(self.flag_save): # Configure depth and color streams
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            self.writerD = cv2.VideoWriter("realsense_depth.avi", fourcc, 60,(640, 480), True)
            self.writerRGB = cv2.VideoWriter("realsense_rgb.avi", fourcc, 60,(640, 480), True)
        else:
            self.writerD = None
            self.writerRGB = None

    def __del__(self):
        print("[INFO] Closing CameraR200 object")

    def close(self):
        if(self.dev is not None): self.dev.stop()
        if(self.srv is not None): self.srv.stop()
        if(self.writerD is not None): self.writerD.release()
        if(self.writerRGB is not None): self.writerRGB.release()

    def apply_configuration(self, lr_exposure=100.0, lr_gain=137.0, ivcam_preset=0):
        try:  # set custom gain/exposure values to obtain good depth image
            custom_options = [(rs_option.RS_OPTION_R200_LR_EXPOSURE, lr_exposure),
                              (rs_option.RS_OPTION_R200_LR_GAIN, lr_gain)]
            self.dev.set_device_options(*zip(*custom_options))
            self.dev.apply_ivcam_preset(ivcam_preset)
        except pyrs.RealsenseError:
            print("[ERROR] CameraR200 --- Could not apply configuration")
            pass

    def get_intrinsics(self, verbose=False):
        intr = self.dev.depth_intrinsics

        fx = intr.fx
        fy = intr.fy
        dscale = self.dev.depth_scale
        if(verbose): print(dscale)

        return intr, dscale

    def get_extrinsics(self, stream_range=(2,3), verbose=False):
        extr = self.dev.get_device_extrinsics(stream_range[0],stream_range[1])
        baseline = extr.translation[0]
        if(verbose): print(baseline)

        return extr, baseline

    def read(self):
        self.dev.wait_for_frames()
        rgb = self.get_rgb_image()
        depth = self.get_depth_image()
        return rgb, depth

    def get_rgb_image(self):
        return self.dev.color

    def get_depth_image(self):
        return self.dev.depth

    def loop(self, verbose=True):
        while True:
            self.count += 1
            if(self.count % 10) == 0:
                now = time.time()
                dt = now - self.last
                fps = 10/dt
                self.fps_smooth = (self.fps_smooth * self.smoothing) + (fps * (1.0-self.smoothing))
                self.last = now
                if(verbose): print("[%.3f] R200 FPS = %.2f" % (now,self.fps_smooth))

            c, d_raw = self.read()

            h,w,ch = c.shape
            d_raw = d_raw * self.dev.depth_scale

            d_raw = d_raw.astype(np.uint8)
            norm_image = cv2.normalize(d_raw, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            norm_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB)
            depth_raw = cv2.applyColorMap(d_raw, cv2.COLORMAP_RAINBOW)

            depth = cv2.medianBlur(d_raw, 5)
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_RAINBOW)
            blank = np.zeros(c.shape, np.uint8)

            cd1 = np.concatenate((c, norm_image), axis=1)
            cd2 = np.concatenate((depth_raw,depth), axis=1)
            cd = np.concatenate((cd1, cd2), axis=0)

            cv2.putText(cd, str(self.fps_smooth)[:4], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
