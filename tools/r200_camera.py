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
        if(self.dev is not None): self.dev.stop()
        if(self.srv is not None): self.srv.stop()
        if(self.writerD is not None): self.writerD.release()
        if(self.writerRGB is not None): self.writerRGB.release()

    def close(self):
        self.__del__()

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

        return intr

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
        img = self.dev.color
        return img

    def get_depth_image(self):
        img = self.dev.depth
        return img
