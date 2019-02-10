
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


#===============================
# import time
# import threading
#
# import numpy as np
#
# import vtk
# import vtk.util.numpy_support as vtk_np
#
# import pyrealsense as pyrs
# from pyrealsense.constants import rs_option
#
# serv = pyrs.Service()
# cam = serv.Device(depth_control_preset=1)
# # serv.start()
# custom_options = [(rs_option.RS_OPTION_R200_LR_EXPOSURE, 100.0),
#                 (rs_option.RS_OPTION_R200_LR_GAIN, 137.0)]
# cam.set_device_options(*zip(*custom_options))
#
# # with serv.Device(depth_control_preset=1) as dev:
# #     try:  # set custom gain/exposure values to obtain good depth image
# #         custom_options = [(rs_option.RS_OPTION_R200_LR_EXPOSURE, 100.0),
# #         (rs_option.RS_OPTION_R200_LR_GAIN, 137.0)]
# #         dev.set_device_options(*zip(*custom_options))
# #     except pyrs.RealsenseError:
# #         pass  # options are not available on all devices
# #     # cam = dev
#
# class VTKActorWrapper(object):
#     def __init__(self, nparray):
#         super(VTKActorWrapper, self).__init__()
#
#         self.nparray = nparray
#
#         nCoords = nparray.shape[0]
#         nElem = nparray.shape[1]
#
#         self.verts = vtk.vtkPoints()
#         self.cells = vtk.vtkCellArray()
#         self.scalars = None
#
#         self.pd = vtk.vtkPolyData()
#         self.verts.SetData(vtk_np.numpy_to_vtk(nparray))
#         self.cells_npy = np.vstack([np.ones(nCoords,dtype=np.int64),
#                                np.arange(nCoords,dtype=np.int64)]).T.flatten()
#         self.cells.SetCells(nCoords,vtk_np.numpy_to_vtkIdTypeArray(self.cells_npy))
#         self.pd.SetPoints(self.verts)
#         self.pd.SetVerts(self.cells)
#
#         self.mapper = vtk.vtkPolyDataMapper()
#         self.mapper.SetInputDataObject(self.pd)
#
#         self.actor = vtk.vtkActor()
#         self.actor.SetMapper(self.mapper)
#         self.actor.GetProperty().SetRepresentationToPoints()
#         self.actor.GetProperty().SetColor(0.0,1.0,0.0)
#
#     def update(self, threadLock, update_on):
#         thread = threading.Thread(target=self.update_actor, args=(threadLock, update_on))
#         thread.start()
#
#     def update_actor(self, threadLock, update_on):
#         while (update_on.is_set()):
#             time.sleep(0.01)
#             threadLock.acquire()
#             cam.wait_for_frames()
#             self.nparray[:] = cam.points.reshape(-1,3)
#             self.pd.Modified()
#             threadLock.release()
#
#
# class VTKVisualisation(object):
#     def __init__(self, threadLock, actorWrapper, axis=True,):
#         super(VTKVisualisation, self).__init__()
#
#         self.threadLock = threadLock
#
#         self.ren = vtk.vtkRenderer()
#         self.ren.AddActor(actorWrapper.actor)
#
#         self.axesActor = vtk.vtkAxesActor()
#         self.axesActor.AxisLabelsOff()
#         self.axesActor.SetTotalLength(1, 1, 1)
#         self.ren.AddActor(self.axesActor)
#
#         self.renWin = vtk.vtkRenderWindow()
#         self.renWin.AddRenderer(self.ren)
#
#         ## IREN
#         self.iren = vtk.vtkRenderWindowInteractor()
#         self.iren.SetRenderWindow(self.renWin)
#         self.iren.Initialize()
#
#         self.style = vtk.vtkInteractorStyleTrackballCamera()
#         self.iren.SetInteractorStyle(self.style)
#
#         self.iren.AddObserver("TimerEvent", self.update_visualisation)
#         dt = 30 # ms
#         timer_id = self.iren.CreateRepeatingTimer(dt)
#
#     def update_visualisation(self, obj=None, event=None):
#         time.sleep(0.01)
#         self.threadLock.acquire()
#         self.ren.GetRenderWindow().Render()
#         self.threadLock.release()
#
#
# def main():
#     update_on = threading.Event()
#     update_on.set()
#
#     threadLock = threading.Lock()
#
#     cam.wait_for_frames()
#     pc = cam.points.reshape(-1,3)
#     actorWrapper = VTKActorWrapper(pc)
#     actorWrapper.update(threadLock, update_on)
#
#     viz = VTKVisualisation(threadLock, actorWrapper)
#     viz.iren.Start()
#     update_on.clear()
#
#
# main()
# cam.stop()
# serv.stop()
